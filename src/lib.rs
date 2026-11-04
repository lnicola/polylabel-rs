#![doc(
    html_logo_url = "https://cdn.rawgit.com/urschrei/polylabel-rs/7a07336e85572eb5faaf0657c2383d7de5620cd8/ell.svg",
    html_root_url = "https://docs.rs/polylabel-rs/"
)]
//! This crate provides a Rust implementation of the [Polylabel](https://github.com/mapbox/polylabel) algorithm
//! for finding the optimum position of a polygon label.
//!
//! ffi bindings are provided: enable the `ffi` and `headers` features when building the crate.
use geo_traits::{CoordTrait, LineStringTrait, PolygonTrait};
use geo_types::{Coord, CoordFloat, Line, Point, Rect};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::ops::{Deref, DerefMut};

pub mod errors;
use errors::PolylabelError;

#[cfg(not(target_arch = "wasm32"))]
#[cfg(feature = "ffi")]
mod ffi;
#[cfg(not(target_arch = "wasm32"))]
#[cfg(feature = "ffi")]
pub use crate::ffi::{Array, Position, WrapperArray, polylabel_ffi};

/// Represention of a Quadtree node's cells. A node contains four Qcells.
#[derive(Debug, Copy, Clone)]
struct Qcell<T>
where
    T: CoordFloat,
{
    // The cell's centroid
    centroid: Point<T>,
    // Half of the parent node's extent
    half_extent: T,
    // Distance from centroid to polygon
    distance: T,
    // Maximum distance to polygon within a cell
    max_distance: T,
}

impl<T> Qcell<T>
where
    T: CoordFloat,
{
    fn new(centroid: Point<T>, half_extent: T, prepared: &PreparedPolygon<T>) -> Self {
        let two = T::one() + T::one();
        let distance = signed_distance(centroid, prepared);
        let max_distance = distance + half_extent * two.sqrt();
        Self {
            centroid,
            half_extent,
            distance,
            max_distance,
        }
    }
}

impl<T> Ord for Qcell<T>
where
    T: CoordFloat,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.max_distance.partial_cmp(&other.max_distance).unwrap()
    }
}
impl<T> PartialOrd for Qcell<T>
where
    T: CoordFloat,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<T> Eq for Qcell<T> where T: CoordFloat {}
impl<T> PartialEq for Qcell<T>
where
    T: CoordFloat,
{
    fn eq(&self, other: &Self) -> bool
    where
        T: CoordFloat,
    {
        self.max_distance == other.max_distance
    }
}

/// Number of consecutive edges grouped under a single bounding box for block-skip
const BLOCK_SIZE: usize = 32;

/// A polygon's rings pre-decomposed into edges, with the edges grouped into
/// fixed-size blocks each carrying a bounding box. Built once per `polylabel`
/// call so the per-cell distance scan can skip whole blocks of edges in O(1).
struct PreparedPolygon<T>
where
    T: CoordFloat,
{
    rings: Vec<PreparedRing<T>>,
}

/// A single ring's edges and the bounding box of each consecutive block of them
struct PreparedRing<T>
where
    T: CoordFloat,
{
    lines: Vec<Line<T>>,
    blocks: Vec<Rect<T>>,
}

impl<T> PreparedPolygon<T>
where
    T: CoordFloat,
{
    fn new(polygon: &impl PolygonTrait<T = T>) -> Self {
        let rings = polygon
            .exterior()
            .into_iter()
            .chain(polygon.interiors())
            .map(|ring| PreparedRing::new(&ring))
            .collect();
        Self { rings }
    }

    /// Follows geo's polygon area and area-weighted ring centroid calculations.
    fn area_and_centroid(&self) -> (T, Option<Point<T>>) {
        let Some((exterior, interiors)) = self.rings.split_first() else {
            return (T::zero(), None);
        };
        let exterior_area = exterior.signed_area();
        let mut area = exterior_area.abs();
        let mut interior_centroid: Option<(T, Coord<T>)> = None;
        for ring in interiors {
            let ring_area = ring.signed_area();
            let weight = ring_area.abs();
            area = area - weight;
            if ring_area == T::zero() {
                continue;
            }
            let accumulated = ring.centroid(ring_area) * weight;
            if let Some((total_weight, total_accumulated)) = &mut interior_centroid {
                *total_weight = *total_weight + weight;
                *total_accumulated = *total_accumulated + accumulated;
            } else {
                interior_centroid = Some((weight, accumulated));
            }
        }
        if exterior_area < T::zero() {
            area = -area;
        }
        // polylabel returns early for zero-area polygons, so their line/point
        // centroid fallbacks are unnecessary here.
        if area == T::zero() || exterior_area == T::zero() {
            return (area, None);
        }
        let mut weight = exterior_area.abs();
        let mut accumulated = exterior.centroid(exterior_area) * weight;
        if let Some((interior_weight, interior_accumulated)) = interior_centroid {
            weight = weight - interior_weight;
            accumulated = accumulated - interior_accumulated;
        }
        let centroid = (weight != T::zero()).then(|| Point::from(accumulated / weight));
        (area, centroid)
    }

    fn bounding_rect(&self) -> Option<Rect<T>> {
        let ring = self.rings.first()?;
        (!ring.lines.is_empty()).then(|| block_bbox(&ring.lines))
    }
}

impl<T> PreparedRing<T>
where
    T: CoordFloat,
{
    /// Follows geo's shifted shoelace sum, omitting the first zero determinant.
    fn signed_area(&self) -> T {
        let Some(first) = self.lines.first() else {
            return T::zero();
        };
        let sum = self.lines.iter().skip(1).fold(T::zero(), |sum, line| {
            let shifted = Line::new(line.start - first.start, line.end - first.start);
            sum + shifted.determinant()
        });
        sum / (T::one() + T::one())
    }

    /// Follows geo's nonzero-area ring centroid, before area weighting.
    fn centroid(&self, area: T) -> Coord<T> {
        let shift = self.lines[0].start;
        let accumulated = self.lines.iter().fold(Coord::zero(), |accum, line| {
            let shifted = Line::new(line.start - shift, line.end - shift);
            accum + (shifted.end + shifted.start) * shifted.determinant()
        });
        accumulated / (T::from(6).unwrap() * area) + shift
    }

    fn new(ring: &impl LineStringTrait<T = T>) -> Self {
        let mut coords = ring.coords().map(|c| Coord { x: c.x(), y: c.y() });
        let mut lines = Vec::with_capacity(ring.num_coords());
        if let Some(first) = coords.next() {
            let mut previous = first;
            for coord in coords {
                lines.push(Line::new(previous, coord));
                previous = coord;
            }
            // Trait implementations may expose open rings; close them just as
            // geo_types::Polygon::new does, without allocating a coordinate copy.
            if previous != first {
                lines.push(Line::new(previous, first));
            }
        }
        let blocks = lines.chunks(BLOCK_SIZE).map(block_bbox).collect();
        Self { lines, blocks }
    }
}

/// Bounding box enclosing every endpoint of a block of edges
fn block_bbox<T>(lines: &[Line<T>]) -> Rect<T>
where
    T: CoordFloat,
{
    let mut min_x = T::infinity();
    let mut min_y = T::infinity();
    let mut max_x = T::neg_infinity();
    let mut max_y = T::neg_infinity();
    for line in lines {
        for c in [line.start, line.end] {
            if c.x < min_x {
                min_x = c.x;
            }
            if c.x > max_x {
                max_x = c.x;
            }
            if c.y < min_y {
                min_y = c.y;
            }
            if c.y > max_y {
                max_y = c.y;
            }
        }
    }
    Rect::new(Coord { x: min_x, y: min_y }, Coord { x: max_x, y: max_y })
}

/// Signed distance from a Qcell's centroid to a Polygon's outline
/// Returned value is negative if the point is outside the polygon's exterior ring
///
/// A single pass over the rings accumulates both the even-odd ray-cast parity
/// (inside/outside) and the minimum distance to the outline. Edges are grouped
/// into blocks (see `PreparedPolygon`): a cheap point-to-bounding-box lower
/// bound skips a whole block when it can neither hold a nearer edge nor flip the
/// ray-cast parity. Most blocks of a ring are far from any given cell centre, so
/// this skips the bulk of the edges.
fn signed_distance<T>(point: Point<T>, prepared: &PreparedPolygon<T>) -> T
where
    T: CoordFloat,
{
    let x = point.x();
    let y = point.y();
    let mut inside = false;
    let mut min_distance = T::infinity();

    for ring in &prepared.rings {
        for (block, bbox) in ring.blocks.iter().enumerate() {
            let start = block * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(ring.lines.len());
            let bmin = bbox.min();
            let bmax = bbox.max();

            // lower bound on the distance from the point to any edge in this
            // block: a point-to-bounding-box clamp, zero when inside the box
            let dx = if x < bmin.x {
                bmin.x - x
            } else if x > bmax.x {
                x - bmax.x
            } else {
                T::zero()
            };
            let dy = if y < bmin.y {
                bmin.y - y
            } else if y > bmax.y {
                y - bmax.y
            } else {
                T::zero()
            };
            let skip_dist = dx * dx + dy * dy >= min_distance * min_distance;

            // edges here can only flip parity if the bbox straddles y and
            // extends right of x; otherwise no edge crosses the rightward ray
            let skip_cross = y < bmin.y || y >= bmax.y || x > bmax.x;

            if skip_dist && skip_cross {
                continue;
            }

            for line in &ring.lines[start..end] {
                let a = line.start;
                let b = line.end;

                if !skip_cross
                    && ((a.y > y) != (b.y > y))
                    && (x < (b.x - a.x) * (y - a.y) / (b.y - a.y) + a.x)
                {
                    inside = !inside;
                }

                if !skip_dist {
                    min_distance = min_distance.min(segment_distance(point, line));
                }
            }
        }
    }

    if inside { min_distance } else { -min_distance }
}

/// Euclidean distance to a segment, including repeated vertices.
/// Follows geo's `geo_types::private_utils::line_segment_distance` implementation.
fn segment_distance<T: CoordFloat>(point: Point<T>, line: &Line<T>) -> T {
    let point = point.0;
    let start = line.start;
    let end = line.end;
    if start == end {
        return (start.x - point.x).hypot(start.y - point.y);
    }
    let dx = end.x - start.x;
    let dy = end.y - start.y;
    let d_squared = dx * dx + dy * dy;
    let r = ((point.x - start.x) * dx + (point.y - start.y) * dy) / d_squared;
    if r <= T::zero() {
        return (start.x - point.x).hypot(start.y - point.y);
    }
    if r >= T::one() {
        return (end.x - point.x).hypot(end.y - point.y);
    }
    let s = ((start.y - point.y) * dx - (start.x - point.x) * dy) / d_squared;
    s.abs() * dx.hypot(dy)
}

struct QuadTree<T>(pub BinaryHeap<Qcell<T>>)
where
    T: CoordFloat;

impl<T> Deref for QuadTree<T>
where
    T: CoordFloat,
{
    type Target = BinaryHeap<Qcell<T>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<T> DerefMut for QuadTree<T>
where
    T: CoordFloat,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> QuadTree<T>
where
    T: CoordFloat,
{
    pub fn new(bbox: Rect<T>, half_extent: T, prepared: &PreparedPolygon<T>) -> Self {
        let mut cell_queue: BinaryHeap<Qcell<T>> = BinaryHeap::new();

        let two = T::one() + T::one();
        let cell_size = half_extent * two;

        let nx = (bbox.width() / cell_size).ceil().to_usize();
        let ny = (bbox.height() / cell_size).ceil().to_usize();

        if let (Some(nx), Some(ny)) = (nx, ny) {
            let one = T::one();
            let delta_mid = Coord { x: one, y: one } * half_extent;
            let origin = bbox.min();
            let inital_points = (0..nx)
                .flat_map(|x| (0..ny).map(move |y| (x, y)))
                .filter_map(|(x, y)| Some((T::from(x)?, T::from(y)?)))
                .map(|(x, y)| Coord { x, y } * cell_size)
                .map(|delta_cell| origin + delta_cell + delta_mid)
                .map(Point::from)
                .map(|centroid| Qcell::new(centroid, half_extent, prepared));
            cell_queue.extend(inital_points);
        } else {
            // Do nothing, maybe error instead?
        }

        Self(cell_queue)
    }

    pub fn add_quad(&mut self, cell: &Qcell<T>, half_extent: T, prepared: &PreparedPolygon<T>) {
        let new_cells = [
            (-T::one(), -T::one()),
            (T::one(), -T::one()),
            (-T::one(), T::one()),
            (T::one(), T::one()),
        ]
        .map(|(sign_x, sign_y)| (sign_x * half_extent, sign_y * half_extent))
        .map(|(dx, dy)| Point::new(dx, dy))
        .map(|delta| cell.centroid + delta)
        .map(|centroid| Qcell::new(centroid, half_extent, prepared));
        self.extend(new_cells);
    }
}

/// Calculate a Polygon's ideal label position by calculating its ✨pole of inaccessibility✨
///
/// The calculation uses an [iterative grid-based algorithm](https://github.com/mapbox/polylabel#how-the-algorithm-works).
///
/// Accepts any [`PolygonTrait`] implementation, including borrowed geometry
/// views. Only XY coordinates are used. Open rings are closed implicitly.
/// The result is a [`geo_types::Point`].
///
/// # Examples
///
/// ```
/// use polylabel::polylabel;
/// extern crate geo;
/// use geo::{Point, LineString, Polygon};
/// use geo::prelude::*;
///
/// // An approximate `L` shape
/// let coords = vec![
///    (0.0, 0.0),
///    (4.0, 0.0),
///    (4.0, 1.0),
///    (1.0, 1.0),
///    (1.0, 4.0),
///    (0.0, 4.0),
///    (0.0, 0.0)];
///
/// let poly = Polygon::new(coords.into(), vec![]);
///
/// // Its centroid lies outside the polygon
/// assert_eq!(poly.centroid().unwrap(), Point::new(1.3571428571428572, 1.3571428571428572));
///
/// let label_position = polylabel(&poly, &0.1).unwrap();
/// // Optimum label position is inside the polygon
/// assert_eq!(label_position, Point::new(0.5625, 0.5625));
/// ```
///
/// # Errors
///
/// The tolerance must be a positive number: zero, negative, or NaN tolerances
/// return [`PolylabelError::InvalidTolerance`], since the refinement loop
/// cannot be guaranteed to terminate for them.
pub fn polylabel<T>(
    polygon: &impl PolygonTrait<T = T>,
    tolerance: &T,
) -> Result<Point<T>, PolylabelError>
where
    T: CoordFloat,
{
    // A non-positive or NaN tolerance can subdivide forever: cells at the pole
    // keep max_distance - best_distance positive (or the comparison is always
    // false for NaN), so the queue grows without bound
    if tolerance.is_nan() || *tolerance <= T::zero() {
        return Err(PolylabelError::InvalidTolerance);
    }

    // Decompose rings into blocks of edges once, up front. All calculations
    // consume this prepared representation rather than an owned input polygon.
    let prepared = PreparedPolygon::new(polygon);
    let (area, centroid) = prepared.area_and_centroid();

    // special case for degenerate polygons
    if area == T::zero() {
        return Ok(Point::new(T::zero(), T::zero()));
    }

    let bbox = prepared
        .bounding_rect()
        .ok_or(PolylabelError::RectCalculation)?;
    let cell_size = bbox.width().min(bbox.height());
    // Special case for degenerate polygons
    if cell_size == T::zero() {
        return Ok(Point::from(bbox.min()));
    }

    let two = T::one() + T::one();
    let mut half_extent = cell_size / two;

    // initial best guess using centroid
    let centroid = centroid.ok_or(PolylabelError::CentroidCalculation)?;
    let centroid_cell = Qcell::new(centroid, T::zero(), &prepared);

    // special case guess for rectangular polygons
    let bbox_cell = Qcell::new(bbox.center().into(), T::zero(), &prepared);

    // deciding which initial guess was better
    let mut best_cell = if bbox_cell.distance > centroid_cell.distance {
        bbox_cell
    } else {
        centroid_cell
    };

    // setup priority queue
    let mut cell_queue = QuadTree::<T>::new(bbox, half_extent, &prepared);

    // Now try to find better solutions
    while let Some(cell) = cell_queue.pop() {
        // Update the best cell if we find a cell with greater distance
        if cell.distance > best_cell.distance {
            best_cell = Qcell { ..cell };
        }

        // Bail out of this iteration if we can't find a better solution
        if cell.max_distance - best_cell.distance <= *tolerance {
            continue;
        }

        // Otherwise, add a new quadtree node and start again
        half_extent = cell.half_extent / two;
        cell_queue.add_quad(&cell, half_extent, &prepared);
    }

    // We've exhausted the queue, so return the best solution we've found
    Ok(best_cell.centroid)
}

#[cfg(test)]
mod tests {
    use super::{Qcell, polylabel};
    use geo::prelude::*;
    use geo::{LineString, Point, Polygon};
    use std::collections::BinaryHeap;

    #[test]
    fn prepared_area_and_centroid_match_geo_with_holes_and_large_offsets() {
        use super::PreparedPolygon;
        for offset in [0., 1e8] {
            for winding in 0..8 {
                let ring = |points: &[(f64, f64)], reverse: bool| {
                    let mut points: Vec<_> = points
                        .iter()
                        .map(|&(x, y)| (x + offset, y + offset))
                        .collect();
                    if reverse {
                        points.reverse();
                    }
                    LineString::from(points)
                };
                let polygon = Polygon::new(
                    ring(
                        &[(0.1, -0.2), (20.3, 0.4), (18.2, 21.8), (-0.4, 19.7)],
                        winding & 1 != 0,
                    ),
                    vec![
                        ring(
                            &[(1.1, 1.2), (1.3, 3.4), (4.5, 3.6), (4.7, 1.8)],
                            winding & 2 != 0,
                        ),
                        ring(
                            &[(10.1, 10.2), (10.3, 13.4), (14.5, 13.6), (14.7, 10.8)],
                            winding & 4 != 0,
                        ),
                        ring(&[(5., 5.), (6., 6.), (7., 7.)], false),
                    ],
                );
                let (area, centroid) = PreparedPolygon::new(&polygon).area_and_centroid();
                assert_eq!(area, polygon.signed_area());
                assert_eq!(centroid, polygon.centroid());
            }
        }
    }

    #[test]
    fn segment_distances_match_geo_including_repeated_vertices() {
        use super::segment_distance;
        use geo::{Euclidean, Line};
        for (start, end) in [((1., 2.), (4., 6.)), ((1., 2.), (1., 2.))] {
            let line: Line<f64> = Line::new(start, end);
            for point in [Point::new(0., 0.), Point::new(2., 3.), Point::new(10., 10.)] {
                let expected = Euclidean.distance(&point, &line);
                assert!((segment_distance(point, &line) - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn segment_distance_uses_exact_endpoints_when_projection_is_clamped() {
        let line = geo::Line::new((1e16, 0.), (1., 0.));
        assert_eq!(super::segment_distance(Point::new(0., 0.), &line), 1.);
    }

    #[test]
    fn coarse_tolerance_keeps_the_better_initial_guess() {
        let polygon: Polygon<f64> =
            Polygon::new(vec![(0., 0.), (10., 0.), (0., 10.)].into(), vec![]);
        // The bounding-box center is on the boundary; the centroid is interior.
        let anchor = polylabel(&polygon, &10.).unwrap();
        let centroid = polygon.centroid().unwrap();
        assert!((anchor.x() - centroid.x()).abs() < 1e-12);
        assert!((anchor.y() - centroid.y()).abs() < 1e-12);
    }

    #[test]
    // polygons are those used in Shapely's tests
    fn test_polylabel() {
        let coords = include!("../tests/fixtures/poly1.rs");
        let poly = Polygon::new(coords.into(), vec![]);
        let res = polylabel(&poly, &10.000).unwrap();
        assert_eq!(
            res,
            Point::new(59.356_155_563_645_69, 121.839_196_297_464_35)
        );
    }
    #[test]
    // does a concave polygon contain the calculated point?
    // relevant because the centroid lies outside the polygon in this case
    fn test_concave() {
        let coords = include!("../tests/fixtures/poly2.rs");
        let poly = Polygon::new(coords.into(), vec![]);
        let res = polylabel(&poly, &1.0).unwrap();
        assert!(poly.contains(&res));
    }
    #[test]
    fn test_london() {
        let coords = include!("../tests/fixtures/poly3.rs");
        let poly = Polygon::new(coords.into(), vec![]);
        let res = polylabel(&poly, &0.001).unwrap();
        assert_eq!(
            res,
            Point::new(-0.455_568_164_459_203_56, 51.548_488_882_028_87)
        );
    }
    #[test]
    fn polygon_l_test() {
        // an L shape
        let coords = vec![
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 1.0),
            (1.0, 1.0),
            (1.0, 4.0),
            (0.0, 4.0),
            (0.0, 0.0),
        ];
        let poly = Polygon::new(coords.into(), vec![]);
        let res = polylabel(&poly, &0.10).unwrap();
        assert_eq!(res, Point::new(0.5625, 0.5625));
    }
    #[test]
    fn degenerate_polygon_test() {
        let a_coords = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.0, 0.0)];
        let a_poly = Polygon::new(a_coords.into(), vec![]);
        let a_res = polylabel(&a_poly, &1.0).unwrap();
        assert_eq!(a_res, Point::new(0.0, 0.0));
    }
    #[test]
    fn degenerate_polygon_test_2() {
        let b_coords = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)];
        let b_poly = Polygon::new(b_coords.into(), vec![]);
        let b_res = polylabel(&b_poly, &1.0).unwrap();
        assert_eq!(b_res, Point::new(0.0, 0.0));
    }
    #[test]
    fn polygon_with_hole_test() {
        let outer = vec![(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)];
        let inner = vec![(60.0, 60.0), (60.0, 80.0), (80.0, 80.0), (80.0, 60.0)];
        let hole_poly = Polygon::new(LineString::from(outer), vec![LineString::from(inner)]);
        let hole_res = polylabel(&hole_poly, &1.0).unwrap();
        assert_eq!(hole_res, Point::new(35.15625, 35.15625));
    }
    #[test]
    // Zero, negative, and NaN tolerances must be rejected rather than
    // subdividing forever (see issue #70)
    fn test_invalid_tolerance() {
        use crate::errors::PolylabelError;
        // a non-square rectangle: its pole is a whole segment of tied
        // maximizers, which previously made a zero tolerance loop forever
        let coords = vec![(0.0, 0.0), (5.0, 0.0), (5.0, 1.0), (0.0, 1.0), (0.0, 0.0)];
        let poly = Polygon::new(coords.into(), vec![]);
        for tol in [0.0, -1.0, f64::NAN] {
            assert_eq!(
                polylabel(&poly, &tol),
                Err(PolylabelError::InvalidTolerance)
            );
        }
    }
    #[test]
    // Is our priority queue behaving as it should?
    fn test_queue() {
        let a = Qcell {
            centroid: Point::new(1.0, 2.0),
            half_extent: 3.0,
            distance: 4.0,
            max_distance: 8.0,
        };
        let b = Qcell {
            centroid: Point::new(1.0, 2.0),
            half_extent: 3.0,
            distance: 4.0,
            max_distance: 7.0,
        };
        let c = Qcell {
            centroid: Point::new(1.0, 2.0),
            half_extent: 3.0,
            distance: 4.0,
            max_distance: 9.0,
        };
        let v = vec![a, b, c];
        let mut q = BinaryHeap::from(v);
        assert_eq!(q.pop().unwrap().max_distance, 9.0);
        assert_eq!(q.pop().unwrap().max_distance, 8.0);
        assert_eq!(q.pop().unwrap().max_distance, 7.0);
    }
}
