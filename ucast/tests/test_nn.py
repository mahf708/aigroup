import pytest
import torch

from ucast.nn import ResBlock, SphereConv2d, UCastUNet, downsample, upsample


def test_circular_padding_makes_convolutions_longitude_equivariant():
    """Rolling the input in longitude must roll the output identically -- the point of wrapping."""
    torch.manual_seed(0)
    conv = SphereConv2d(3, 5, kernel=3, periodic_longitude=True)
    x = torch.randn(2, 3, 9, 12)
    shift = 4
    rolled = conv(torch.roll(x, shifts=shift, dims=-1))
    assert torch.allclose(rolled, torch.roll(conv(x), shifts=shift, dims=-1), atol=1e-6)


def test_non_periodic_convolution_is_not_longitude_equivariant():
    torch.manual_seed(0)
    conv = SphereConv2d(3, 3, kernel=3, periodic_longitude=False)
    x = torch.randn(1, 3, 8, 8)
    rolled = conv(torch.roll(x, 3, dims=-1))
    assert not torch.allclose(rolled, torch.roll(conv(x), 3, dims=-1), atol=1e-4)


def test_latitude_padding_modes_change_the_poles_only():
    torch.manual_seed(0)
    x = torch.randn(1, 2, 6, 8)
    zeros = SphereConv2d(2, 2, kernel=3, latitude_padding="zeros")
    replicate = SphereConv2d(2, 2, kernel=3, latitude_padding="replicate")
    replicate.load_state_dict(zeros.state_dict())
    a, b = zeros(x), replicate(x)
    assert torch.allclose(a[:, :, 1:-1], b[:, :, 1:-1], atol=1e-6)
    assert not torch.allclose(a[:, :, 0], b[:, :, 0], atol=1e-4)


def test_zero_kernel_convolution_is_an_identity():
    conv = SphereConv2d(4, 4, kernel=0)
    x = torch.randn(2, 4, 5, 6)
    assert torch.equal(conv(x), x)
    assert list(conv.parameters()) == []


def test_zero_initialised_convolution_starts_at_zero():
    conv = SphereConv2d(3, 3, kernel=3, init_weight=0.0, init_bias=0.0)
    assert torch.count_nonzero(conv(torch.randn(1, 3, 4, 4))) == 0


def test_resampling_roundtrips_odd_extents():
    x = torch.randn(1, 2, 121, 240)
    small = downsample(x)
    assert tuple(small.shape[-2:]) == (61, 120)  # ceil mode keeps the pole row
    assert tuple(upsample(small, (121, 240)).shape[-2:]) == (121, 240)


def test_resblock_modes_and_attention():
    block = ResBlock(4, 8, mode="down", attention=True, channels_per_head=4, dropout=0.5)
    out = block(torch.randn(2, 4, 8, 16))
    assert tuple(out.shape) == (2, 8, 4, 8)
    up = ResBlock(8, 8, mode="up")
    assert tuple(up(out, size=(9, 17)).shape) == (2, 8, 9, 17)


def test_unet_preserves_the_grid_on_a_non_power_of_two_shape():
    net = UCastUNet(
        in_channels=6, out_channels=3, spatial_shape=(121, 240), model_channels=8, channel_mult=(1, 2, 3),
        num_blocks=1, attn_levels=(-1,), channels_per_head=8,
    )
    assert net.level_sizes == [(121, 240), (61, 120), (31, 60)]
    out = net(torch.randn(1, 6, 121, 240))
    assert tuple(out.shape) == (1, 3, 121, 240)


def test_unet_output_starts_at_zero_so_stage_one_begins_at_persistence():
    net = UCastUNet(in_channels=4, out_channels=2, spatial_shape=(16, 32), model_channels=8, channel_mult=(1, 2),
                    num_blocks=1, attn_levels=())
    net.eval()
    assert torch.count_nonzero(net(torch.randn(2, 4, 16, 32))) == 0


def test_unet_rejects_the_wrong_channel_count():
    net = UCastUNet(in_channels=4, out_channels=2, spatial_shape=(8, 16), model_channels=8, channel_mult=(1,),
                    num_blocks=1, attn_levels=())
    with pytest.raises(ValueError, match="input channels"):
        net(torch.randn(1, 5, 8, 16))


def test_dropout_makes_forward_passes_differ_only_in_training_mode():
    torch.manual_seed(0)
    net = UCastUNet(in_channels=4, out_channels=2, spatial_shape=(16, 32), model_channels=16, channel_mult=(1, 2),
                    num_blocks=1, attn_levels=(), dropout=0.3)
    # At initialisation every residual branch is zero (conv1 and out_conv are zero-initialised), so
    # the model is a pure skip path and dropout provably cannot change anything. Perturb those layers
    # to emulate a trained model.
    with torch.no_grad():
        net.out_conv.weight.normal_(0, 0.1)
        for name, parameter in net.named_parameters():
            if name.endswith("conv1.weight"):
                parameter.normal_(0, 0.1)
    x = torch.randn(1, 4, 16, 32)
    net.eval()
    assert torch.allclose(net(x), net(x))
    net.train()
    assert not torch.allclose(net(x), net(x))


def test_paper_configuration_has_the_published_parameter_count():
    """The paper reports 895M parameters for the 1.5-degree model."""
    with torch.device("meta"):
        net = UCastUNet(
            in_channels=172, out_channels=83, spatial_shape=(121, 240), model_channels=320,
            channel_mult=(1, 2, 3, 4), num_blocks=4, attn_levels=(-2, -1),
        )
    assert 880e6 < net.num_parameters < 910e6
