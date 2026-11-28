"""The V2 commissure rule must not admit arbitrary native skin locations."""
import pytest

from persona_lab.mediapipe_face import SCHEMAS
from persona_lab.surface_correspondences import NativeLocation, required_groups


@pytest.mark.parametrize('slot,index', [(61,48), (291,54), (78,60), (308,64)])
def test_exact_commissure_rule(slot, index):
    point = SCHEMAS['mediapipe-face-478-dfl'].points[slot]
    assert required_groups(point, NativeLocation('landmark', index)) == (('skin',),)
    assert required_groups(point, NativeLocation('landmark', 30)) == (('upper_lip',), ('lower_lip',))
    assert required_groups(point, NativeLocation('triangle', 0, (1.,0.,0.))) == (('upper_lip',), ('lower_lip',))
