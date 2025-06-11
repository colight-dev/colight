# colight.scene3d {: .api .api-title }

### Scene {: .api .api-member }

A 3D scene visualization component using WebGPU.

This class creates an interactive 3D scene that can contain multiple types of components:

- Point clouds
- Ellipsoids
- Ellipsoid bounds (wireframe)
- Cuboids

The visualization supports:

- Orbit camera control (left mouse drag)
- Pan camera control (shift + left mouse drag or middle mouse drag)
- Zoom control (mouse wheel)
- Component hover highlighting
- Component click selection
- Optional FPS display (set controls=['fps'])



### PointCloud {: .api .api-member }

Create a point cloud element.

Parameters
{: .api .api-section }


- `centers` (ArrayLike): Nx3 array of point centers or flattened array

- `colors` (Optional[ArrayLike]): Nx3 array of RGB colors or flattened array (optional)

- `color` (Optional[ArrayLike]): Default RGB color [r,g,b] for all points if colors not provided

- `sizes` (Optional[ArrayLike]): N array of point sizes or flattened array (optional)

- `size` (Optional[NumberLike]): Default size for all points if sizes not provided

- `alphas` (Optional[ArrayLike]): Array of alpha values per point (optional)

- `alpha` (Optional[NumberLike]): Default alpha value for all points if alphas not provided

- `**kwargs` (Any): Additional arguments like decorations, onHover, onClick



### Ellipsoid {: .api .api-member }

Create an ellipsoid element.

Parameters
{: .api .api-section }


- `centers` (ArrayLike): Nx3 array of ellipsoid centers or flattened array

- `half_sizes` (Optional[ArrayLike]): Nx3 array of half_sizes (x,y,z) or flattened array (optional)

- `half_size` (Optional[Union[NumberLike, ArrayLike]]): Default half_size (sphere) or [x,y,z] half_sizes (ellipsoid) if half_sizes not provided

- `quaternions` (Optional[ArrayLike]): Nx4 array of orientation quaternions [x,y,z,w] (optional)

- `quaternion` (Optional[ArrayLike]): Default orientation quaternion [x,y,z,w] if quaternions not provided

- `colors` (Optional[ArrayLike]): Nx3 array of RGB colors or flattened array (optional)

- `color` (Optional[ArrayLike]): Default RGB color [r,g,b] for all ellipsoids if colors not provided

- `alphas` (Optional[ArrayLike]): Array of alpha values per ellipsoid (optional)

- `alpha` (Optional[NumberLike]): Default alpha value for all ellipsoids if alphas not provided

- `fill_mode` (str): How the shape is drawn. One of:

    - "Solid": Filled surface with solid color

    - "MajorWireframe": Three axis-aligned ellipse cross-sections

- `**kwargs` (Any): Additional arguments like decorations, onHover, onClick



### Cuboid {: .api .api-member }

Create a cuboid element.

Parameters
{: .api .api-section }


- `centers` (ArrayLike): Nx3 array of cuboid centers or flattened array

- `half_sizes` (Optional[ArrayLike]): Nx3 array of half sizes (width,height,depth) or flattened array (optional)

- `half_size` (Optional[Union[ArrayLike, NumberLike]]): Default half size [w,h,d] for all cuboids if half_sizes not provided

- `quaternions` (Optional[ArrayLike]): Nx4 array of orientation quaternions [x,y,z,w] (optional)

- `quaternion` (Optional[ArrayLike]): Default orientation quaternion [x,y,z,w] if quaternions not provided

- `colors` (Optional[ArrayLike]): Nx3 array of RGB colors or flattened array (optional)

- `color` (Optional[ArrayLike]): Default RGB color [r,g,b] for all cuboids if colors not provided

- `alphas` (Optional[ArrayLike]): Array of alpha values per cuboid (optional)

- `alpha` (Optional[NumberLike]): Default alpha value for all cuboids if alphas not provided

- `**kwargs` (Any): Additional arguments like decorations, onHover, onClick



### LineBeams {: .api .api-member }

Create a line beams element.

Parameters
{: .api .api-section }


- `points` (ArrayLike): Array of quadruples [x,y,z,i, x,y,z,i, ...] where points sharing the same i value are connected in sequence

- `color` (Optional[ArrayLike]): Default RGB color [r,g,b] for all beams if colors not provided

- `size` (Optional[NumberLike]): Default size for all beams if sizes not provided

- `colors` (Optional[ArrayLike]): Array of RGB colors per line (optional)

- `sizes` (Optional[ArrayLike]): Array of sizes per line (optional)

- `alpha` (Optional[NumberLike]): Default alpha value for all beams if alphas not provided

- `alphas` (Optional[ArrayLike]): Array of alpha values per line (optional)

- `**kwargs` (Any): Additional arguments like onHover, onClick

Returns
{: .api .api-section }


- A LineBeams scene component that renders connected beam segments. (SceneComponent)

- Points are connected in sequence within groups sharing the same i value. (SceneComponent)



### deco {: .api .api-member }

Create a decoration for scene components.

Parameters
{: .api .api-section }


- `indexes` (Union[int, np.integer, ArrayLike]): Single index or list of indices to decorate

- `color` (Optional[ArrayLike]): Optional RGB color override [r,g,b]

- `alpha` (Optional[NumberLike]): Optional opacity value (0-1)

- `scale` (Optional[NumberLike]): Optional scale factor

Returns
{: .api .api-section }


- Dictionary containing decoration settings (Decoration)
