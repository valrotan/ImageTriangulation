def write_obj_simple(vertices, faces, filename="out.obj"):
    """
    vertices - ordered array of vertices, where each vertex has 3 elements
               corresponding to x, y, z
    faces - set of faces, where each face is an array of vertex indexes, starting at 0
    filename - output filename
    """

    with open(filename, "w+") as out_file:
        for vert in vertices:
            print('v %s' % ' '.join(map(str, vert)), file=out_file)
        print('\n\n', file=out_file)
        for face in faces:
            print('f %s' % ' '.join(map(lambda i : str(i + 1), face)), file=out_file)

class Face:
    def __init__(self, verts=[], material=None):
        """
        verts - set of vertex indexes
        material - index of material used, None to be the same as the previous triangle
        """
        self.verts = verts
        self.material = material

def write_obj(vertices, faces, materials=None, filename="out.obj"):
    """
    vertices - ordered array of vertices, where each vertex has 3 elements
               corresponding to x, y, z
    faces - set of faces, where each face is a member of the Face class above
    materials - set of materials where each material is a set of rgb float values [0-1]
    filename - output filename
    """

    mtl_filename = filename.replace("obj", "mtl")

    with open(filename, "w+") as out_file:
        if materials is not None:
            print("mtllib", mtl_filename, file=out_file)

        for vert in vertices:
            print('v %s' % ' '.join(map(str, vert)), file=out_file)

        print('\n\n', file=out_file)

        last_mtl = None
        for face in faces:
            if face.material is not None and last_mtl != face.material:
                last_mtl = face.material
                print("usemtl m%d" % last_mtl, file=out_file)
            print('f %s' % ' '.join(map(lambda i : str(i + 1), face.verts)), file=out_file)

    if materials is not None:
        with open(mtl_filename, "w+") as out_file:
            for i, mat in enumerate(materials):
                print('newmtl m%d' % i, file=out_file)
                print('Ka %s' % ' '.join(map(lambda i : str(i), mat)), file=out_file)
                print(file=out_file)
