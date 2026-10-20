# Third-party data, models, and software

The MIT license in this repository covers the MVCAP generation code and its
documentation. It does not grant rights to separately obtained datasets, 3D
models, body-model parameters, motion recordings, or external software.

## L2: ShapeNetCore

Obtain ShapeNetCore from the [official ShapeNet organization](https://huggingface.co/ShapeNet)
and review the [ShapeNet terms](https://huggingface.co/spaces/ShapeNet/README/blob/main/TERMS.md).
The selection script expects category ZIP archives containing
`<synset>/<model_id>/models/model_normalized.obj` and associated resources.
Downloaded ZIP archives and extracted meshes are local inputs and are not
distributed in this repository.

## L3: BABEL, motion recordings, and SMPL

- Obtain action annotations from [BABEL](https://babel.is.tue.mpg.de/) under its
  [license](https://babel.is.tue.mpg.de/license.html).
- The loader expects the CMU subset in the AMASS-style NPZ representation.
  Obtain motion files from the appropriate [AMASS](https://amass.is.tue.mpg.de/)
  source and review the applicable terms. The [CMU motion database](https://mocap.cs.cmu.edu/)
  is the original recording source; original CMU files and AMASS conversions
  should not be treated as having interchangeable licenses or file formats.
- Obtain a neutral [SMPL body model](https://smpl.is.tue.mpg.de/) from its
  official provider under the relevant model license. Model parameters and
  example model distributions are not included here.
- The `smplx` package is installed separately from
  [its official package](https://pypi.org/project/smplx/).

BABEL has explicit redistribution restrictions. Do not add downloaded BABEL
files, SMPL parameters, or motion arrays to this Git repository. Any sharing
of generated stimuli or derived annotations needs its own review against the
applicable upstream terms; the code license does not resolve those rights.

## Research visualizations

The small [example gallery](docs/examples/README.md) contains rendered MVCAP
stimuli with source acknowledgments and sample provenance. It does not include
downloaded source datasets, motion arrays, meshes, or body-model parameters.
The MIT code license does not replace the terms of the underlying resources.

## Software dependencies

Dependencies are installed from their own distributions and retain their own
licenses, including NumPy, OpenCV, Pillow, trimesh, pyrender, PyOpenGL, PyTorch,
and smplx. FFmpeg is an external executable. The repository does not vendor
their source code or binaries.

Please also cite the source datasets and body model when they are used in
your research, following the citation instructions on their official sites.
