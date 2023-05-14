=============================================
Spherical to Cartesian Coordinates Conversion
=============================================

In spherical coordinates, a point in space is described by three values: the radial distance of that point from the origin, the inclination angle from the positive z-axis, and the azimuth angle from the positive x-axis and the intersection between the plane passing through the point and the origin perpendicular to the z-axis.

The spherical coordinates are usually represented as `(r, theta, phi)`, where:

- `r` is the radial distance,
- `theta` is the inclination angle (ranging from 0 to pi), and
- `phi` is the azimuth angle (ranging from 0 to 2pi).

The conversion from spherical to Cartesian coordinates is given by the following equations:

- `x = r * cos(theta)`
- `y = r * sin(theta) * cos(phi)`
- `z = r * sin(theta) * sin(phi)`

For example, if we have a point in spherical coordinates as `(1, pi/2, 0)`, the conversion to Cartesian coordinates would be:

- `x = r * cos(pi/2) = 0`
- `y = r * sin(pi/2) * cos(0) = 1`
- `z = r * sin(pi/2) * sin(0) = 0`

So the Cartesian coordinates for this point would be `(0, 1, 0)`.

It's important to note that in a 2D space (or when `r` and `theta` are defined, but `phi` is not), the conversion simplifies to:

- `x = r * cos(theta)`
- `y = r * sin(theta)`

The `spherical_to_cartesian` function implements these conversions.
For the multi-dimension case see the article here
`N-sphere <https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates>`__.