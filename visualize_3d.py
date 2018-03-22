import numpy as np
from skimage import measure
from plotly import figure_factory as ff
from plotly.offline import plot, iplot
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import vtk
from scipy.ndimage import median_filter
from stl import mesh


def make_mesh(image, threshold=0.5, step_size=1):

    vertices, faces, norm, val = measure.marching_cubes(image, threshold, step_size=step_size,
                                                        gradient_direction='ascent', allow_degenerate=True)
    return vertices, faces, norm


def plotly_3d(vertices, faces):
    """Creates fast, but not so high quality plot with plotly as html embedded javascript."""
    x, y, z = zip(*vertices)

    # Make the colormap single color since the axes are positional not intensity.
    #    colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']
    colormap = ['rgb(236, 236, 212)', 'rgb(236, 236, 212)']

    fig = ff.create_trisurf(x=x,
                            y=y,
                            z=z,
                            plot_edges=False,
                            colormap=colormap,
                            simplices=faces,
                            backgroundcolor='rgb(64, 64, 64)',
                            title="Interactive Visualization")
    plot(fig)
    # if in jupyter notebook:
    # init_notebook_mode(connected=True)
    # iplot(fig)


def plt_3d(vertices, faces):
    """Creates slow, but high quality plot with matplotlib."""

    x, y, z = zip(*vertices)
    fig = plt.figure(figsize=(10, 10))
    ax = Axes3D(fig)

    # Fancy indexing: `vertices[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(vertices[faces], linewidths=0.05, alpha=1)
    face_color = [1, 1, 0.9]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))
    ax.set_axis_bgcolor((0.7, 0.7, 0.7))
    plt.show()


def vtk_volume_rendering(data):

    # for some reason, vtk only allows same cubic dimensions
    if data.shape[0] != data.shape[1] or data.shape[0] != data.shape[2]:
        new_shape = (np.max(data.shape), np.max(data.shape), np.max(data.shape))
        data_new = np.zeros(new_shape)
        data_new[:data.shape[0], :data.shape[1], :data.shape[2]] = data
        data = data_new

    # the data must be reduced to unsigned 8 bit or 16 bit integers.
    data = (data * 255).astype(np.uint8)

    data_importer = vtk.vtkImageImport()
    data_string = data.tostring()
    data_importer.CopyImportVoidPointer(data_string, len(data_string))
    data_importer.SetDataScalarTypeToUnsignedChar()
    data_importer.SetNumberOfScalarComponents(1)
    data_importer.SetDataExtent(0, data.shape[0]-1, 0, data.shape[1]-1, 0, data.shape[2]-1)
    data_importer.SetWholeExtent(0, data.shape[0]-1, 0, data.shape[1]-1, 0, data.shape[2]-1)

    a_renderer = vtk.vtkRenderer()
    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(a_renderer)

    # Create transfer mapping scalar value to opacity
    opacity_transfer_function = vtk.vtkPiecewiseFunction()
    opacity_transfer_function.AddPoint(0, 0.0)
    opacity_transfer_function.AddPoint(70, 0.0)
    opacity_transfer_function.AddPoint(75, 0.135)
    opacity_transfer_function.AddPoint(255, 1.0)

    # Create transfer mapping scalar value to color
    color_transfer_function = vtk.vtkColorTransferFunction()
    color_transfer_function.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
    color_transfer_function.AddRGBPoint(128.0, 0.3, 0.6, 0.6)

    # The property describes how the data will look
    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(color_transfer_function)
    volume_property.SetScalarOpacity(opacity_transfer_function)
    volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()

    # The mapper / ray cast function know how to render the data
    volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
    volume_mapper.SetBlendModeToComposite()
    volume_mapper.SetInputConnection(data_importer.GetOutputPort())

    # The volume holds the mapper and the property and
    # can be used to position/orient the volume
    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    a_renderer.AddVolume(volume)

    # add axes
    axes = vtk.vtkAxesActor()
    transform = vtk.vtkTransform()
    transform.Scale(50.0, 50.0, 50.0)
    axes.SetUserTransform(transform)
    a_renderer.AddActor(axes)

    # set camera
    a_camera = vtk.vtkCamera()
    a_camera.SetViewUp(0, 0, -1)
    a_camera.SetPosition(0, 1, 0)
    a_camera.SetFocalPoint(0, 0, 0)
    a_camera.ComputeViewPlaneNormal()

    a_renderer.SetActiveCamera(a_camera)
    a_renderer.ResetCamera()

    a_renderer.SetBackground(1.0, 1.0, 1.0)
    ren_win.SetSize(1024, 768)

    a_renderer.ResetCameraClippingRange()

    # Interact with the data.
    render_interactor = vtk.vtkRenderWindowInteractor()
    render_interactor.SetRenderWindow(ren_win)
    render_interactor.Initialize()
    ren_win.Render()

    interactor_style = vtk.vtkInteractorStyleTrackballCamera()
    render_interactor.SetInteractorStyle(interactor_style)
    render_interactor.Start()


def save_stl(vertices, faces, normals, filename: str = 'out.stl'):
    # create mesh
    out_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

    for i, f in enumerate(faces):
        for j in range(3):
            out_mesh.vectors[i][j] = vertices[f[j], :]

    print(out_mesh.normals.shape)

    out_mesh.save(filename)


def save_ply_ascii(vertices, faces, normals, filename: str = 'out.ply'):

    with open(filename, 'w') as file:
        file.write("""ply
format ascii 1.0
comment Max-Heinrich Laves
element vertex {}
property float x
property float y
property float z
property float nx
property float ny
property float nz
element face {}
property list uchar int vertex_indices
end_header\n""".format(vertices.shape[0], faces.shape[0]))
        for v, n in zip(vertices, normals):
            file.write(str(v[0]) + ' ' + str(v[1]) + ' ' + str(v[2]) + ' '
                       + str(n[0]) + ' ' + str(n[1]) + ' ' + str(n[2]) + '\n')

        for f in faces:
            file.write(str(3) + ' ' + str(f[0]) + ' ' + str(f[1]) + ' ' + str(f[2]) + '\n')


def save_ply_binary(vertices, faces, normals, filename: str = 'out.ply'):

    with open(filename, 'wb') as file:
        file.write("""ply
format binary_little_endian 1.0
comment by imes4d
element vertex {}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
property uchar alpha
element face {}
property list uchar int vertex_indices
end_header\n""".format(vertices.shape[0], faces.shape[0]).encode('ascii'))

        for v, n in zip(vertices, normals):
            np.array(v, dtype=np.float32).tofile(file)
            np.array(n, dtype=np.float32).tofile(file)
            np.array([227, 218, 201, 255], dtype=np.ubyte).tofile(file)

        for f in faces:
            np.array([3], dtype=np.ubyte).tofile(file)
            np.array(f, dtype=np.int32).tofile(file)


if __name__ == '__main__':
    A = np.load('stitched_total.npz')
    a = A[A.files[0]].astype(np.float32)

    a = median_filter(a, size=5)
    v, f, n = make_mesh(a, 0.25, 2)

    #plotly_3d(v, f)
    #vtk_volume_rendering(a)

    # save as stl
    #save_stl(v, f, n, 'stitched_total.stl')
    save_ply_binary(v, f, n, 'stitched_bin.ply')
