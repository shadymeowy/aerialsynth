//! Geodesy and Web-Mercator XYZ tile math for the aerial-odometry simulator.
//!
//! All computations are carried out in `f64`. Conventions (shared by every module):
//!
//! * **Angles are radians** in the core API (`_deg` helpers exist where noted).
//! * **Geodetic** coordinates are latitude/longitude on the reference ellipsoid
//!   (default [`Ellipsoid::WGS84`]) and height `h` in meters *above the ellipsoid*
//!   (not above the geoid / MSL).
//! * **ECEF** is the Earth-centred, Earth-fixed Cartesian frame in meters
//!   (x towards lat=0/lon=0, z towards the north pole).
//! * **ENU** = east/north/up and **NED** = north/east/down local tangent frames whose
//!   "up" axis is the geodetic surface normal. Naming and semantics follow
//!   [pymap3d](https://github.com/geospace-code/pymap3d), e.g. [`geodetic2enu`].
//! * **AER** = azimuth (clockwise from north, in `[0, 2π)`), elevation (above the local
//!   horizontal plane) and slant range in meters.
//! * **Body frame** is FRD (x forward, y right, z down); see [`attitude`].
//! * **Tiles** are Google/OSM "slippy map" XYZ tiles in Web Mercator (EPSG:3857); see
//!   [`tiles`].

pub mod attitude;
pub mod ellipsoid;
pub mod frames;
pub mod tiles;

pub use attitude::{
    body2ecef_to_body2ned, body2ned_to_body2ecef, dmat3_to_quat, euler_zyx_to_quat,
    quat_ecef_from_ned, quat_to_euler_zyx,
};
pub use ellipsoid::Ellipsoid;
pub use frames::{
    aer2ecef, aer2enu, aer2geodetic, aer2ned, ecef2aer, ecef2enu, ecef2enuv, ecef2geodetic,
    ecef2ned, ecef2nedv, enu2aer, enu2ecef, enu2ecefv, enu2geodetic, geodetic2aer, geodetic2ecef,
    geodetic2enu, geodetic2ned, haversine_distance, horizon_distance, ned2aer, ned2ecef, ned2ecefv,
    ned2geodetic, ray_ellipsoid, rot_ecef2enu, rot_ecef2ned, up_vector, Geodetic, LocalConvention,
    LocalFrame,
};
pub use tiles::{
    gsd_ew, gsd_ns, latlon_to_pixel, latlon_to_uv, mercator_xy, mercator_xy_to_latlon,
    pixel_to_latlon, tile_for_latlon, tiles_in_bounds, uv_to_latlon, zoom_for_gsd, zoom_for_gsd_f,
    LatLonBounds, TileId, MAX_MERCATOR_LAT, MAX_MERCATOR_LAT_RAD, MAX_ZOOM, WEB_MERCATOR_R,
};

/// Re-exported glam types used throughout the API.
pub use glam::{DMat3, DQuat, DVec2, DVec3};
