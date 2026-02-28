# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# Option to expand the port at the top. P. Mazerewicz, 2026
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Creating ducts for the port
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.error import BuilderError
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.builders.tools import apply_component_display_options
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import (
    boolean_fragments,
    boolean_fuse,
    extrude_shape,
    make_polygon,
    offset_wire,
    point_inside_shape,
)
from bluemira.materials import Void

GEOM_TOL = 1e-12

# -----------------------------------------------------------------------------
# UPPER PORT EXPANSION PARAMETERS
# -----------------------------------------------------------------------------
# Percent values are provided in reactor parameters and interpreted relative to:
# - z_max for the depth of the expanded section measured from the top
# - base X span (x_max - x_min) for radial and lateral expansions.

if TYPE_CHECKING:
    from bluemira.base.reactor_config import ConfigParams
    from bluemira.geometry.solid import BluemiraSolid
    from bluemira.geometry.wire import BluemiraWire


@dataclass
class TSUpperPortDuctBuilderParams(ParameterFrame):
    """Thermal shield upper port duct builder Parameter Frame"""

    n_TF: Parameter[int]
    tf_wp_depth: Parameter[float]
    g_ts_tf: Parameter[float]
    tk_ts: Parameter[float]
    g_cr_ts: Parameter[float]
    upper_port_expanded_section_depth_from_top_pct: Parameter[float]
    upper_port_expand_radial_outward_pct: Parameter[float]
    upper_port_expand_radial_inward_pct: Parameter[float]
    upper_port_expand_lateral_pct: Parameter[float]


class TSUpperPortDuctBuilder(Builder):
    """Thermal shield upper port duct builder"""

    params: TSUpperPortDuctBuilderParams
    param_cls: type[TSUpperPortDuctBuilderParams] = TSUpperPortDuctBuilderParams

    TS = "TS"

    def __init__(
        self,
        params: dict | ParameterFrame | ConfigParams | None,
        build_config: dict,
        port_koz: BluemiraFace,
        cryostat_ts_xz: BluemiraWire,
    ):
        super().__init__(params, build_config)
        self.x_min, self.x_max = _base_x_bounds_from_koz_bbox(
            port_koz.bounding_box.x_min,
            port_koz.bounding_box.x_max,
            self.params.upper_port_expand_radial_inward_pct.value,
            self.params.upper_port_expand_radial_outward_pct.value,
        )
        self.z_max = cryostat_ts_xz.bounding_box.z_max + 0.5 * self.params.g_cr_ts.value

        if self.params.tk_ts.value <= 0:
            raise ValueError("Port wall thickness must be > 0")

        self.y_offset = self.params.tf_wp_depth.value + self.params.g_ts_tf.value

    def build(self) -> Component:
        """Build upper port

        Returns
        -------
        :
            The upper port component tree
        """
        xy_face = make_upper_port_xy_face(
            self.params.n_TF.value,
            self.x_min,
            self.x_max,
            self.params.tk_ts.value,
            self.params.tk_ts.value,
            self.y_offset,
        )

        return self.component_tree(
            None, [self.build_xy(xy_face)], self.build_xyz(xy_face)
        )

    def build_xyz(self, xy_face: BluemiraFace) -> PhysicalComponent:
        """
        Build upper port xyz

        Returns
        -------
        :
            The xyz components
        """
        expand_any = any(
            abs(v) > GEOM_TOL
            for v in (
                self.params.upper_port_expanded_section_depth_from_top_pct.value,
                self.params.upper_port_expand_radial_outward_pct.value,
                self.params.upper_port_expand_radial_inward_pct.value,
                self.params.upper_port_expand_lateral_pct.value,
            )
        )

        if not expand_any:
            xy_voidface = BluemiraFace(xy_face.boundary[1])
            xy_outface = BluemiraFace(xy_face.boundary[0])
            port = extrude_shape(xy_face, (0, 0, self.z_max))
            # Add start-cap for future boolean fragmentation help
            cap = extrude_shape(xy_outface, vec=(0, 0, 0.1))
            port = boolean_fuse([port, cap])
            comp = PhysicalComponent(
                self.name,
                port,
                material=self.get_material(self.TS),
            )
            apply_component_display_options(comp, BLUE_PALETTE[self.TS][0])
            void = PhysicalComponent(
                self.name + " voidspace",
                extrude_shape(xy_voidface, (0, 0, self.z_max)),
                material=Void("vacuum"),
            )
            apply_component_display_options(void, color=(0, 0, 0))
            return [comp, void]

        y_expand = _side_expand_m(
            self.x_min,
            self.x_max,
            self.params.upper_port_expand_lateral_pct.value,
        )
        x_min_outer, x_max_outer = _expanded_x_bounds(
            self.x_min,
            self.x_max,
            self.params.upper_port_expand_radial_inward_pct.value,
            self.params.upper_port_expand_radial_outward_pct.value,
        )
        wall_end_tk = self.params.tk_ts.value
        wall_side_tk = self.params.tk_ts.value
        expand_from_top_m = _pct_to_m(
            self.params.upper_port_expanded_section_depth_from_top_pct.value, self.z_max
        )

        xy_base = xy_face
        xy_shelf = _make_upper_port_xy_face_with_shelf(
            self.params.n_TF.value,
            x_min_base=self.x_min,
            x_max_base=self.x_max,
            x_min_outer=x_min_outer,
            x_max_outer=x_max_outer,
            wall_end_tk=wall_end_tk,
            wall_side_tk=wall_side_tk,
            y_offset=self.y_offset,
            y_expand=y_expand,
        )
        if abs(y_expand) > GEOM_TOL:
            xy_top = _make_upper_port_xy_face_y_expanded(
                self.params.n_TF.value,
                x_min_outer,
                x_max_outer,
                wall_end_tk,
                wall_side_tk,
                self.y_offset,
                y_expand,
                allow_outer_trim=False,
                allow_inner_trim=False,
            )
        else:
            xy_top = _make_upper_port_xy_face_no_outer_trim(
                self.params.n_TF.value,
                x_min_outer,
                x_max_outer,
                wall_end_tk,
                wall_side_tk,
                self.y_offset,
            )

        port = _extrude_three_parts(
            xy_base,
            xy_shelf,
            xy_top,
            self.z_max,
            shelf_h=max(wall_end_tk, wall_side_tk),
            expand_from_top_m=expand_from_top_m,
        )

        xy_outface = BluemiraFace(xy_base.boundary[0])
        cap = extrude_shape(xy_outface, vec=(0, 0, 0.1))
        port = boolean_fuse([port, cap])

        void = _extrude_three_parts(
            BluemiraFace(xy_base.boundary[1]),
            BluemiraFace(xy_shelf.boundary[1]),
            BluemiraFace(xy_top.boundary[1]),
            self.z_max,
            shelf_h=max(wall_end_tk, wall_side_tk),
            expand_from_top_m=expand_from_top_m,
        )

        comp = PhysicalComponent(self.name, port, material=self.get_material(self.TS))
        void = PhysicalComponent(
            self.name + " voidspace", void, material=Void("vacuum")
        )
        apply_component_display_options(comp, BLUE_PALETTE[self.TS][0])
        apply_component_display_options(void, color=(0, 0, 0))
        return [comp, void]

    def build_xy(self, face: BluemiraFace) -> PhysicalComponent:
        """
        Build upport port xy face

        Returns
        -------
        :
            The xy component
        """
        comp = PhysicalComponent(self.name, face)
        apply_component_display_options(comp, BLUE_PALETTE[self.TS][0])
        return comp


@dataclass
class TSEquatorialPortDuctBuilderParams(ParameterFrame):
    """Thermal shield upper port duct builder Parameter Frame"""

    n_TF: Parameter[int]
    R_0: Parameter[float]
    tf_wp_depth: Parameter[float]
    g_ts_tf: Parameter[float]
    g_vv_ts: Parameter[float]
    tk_ts: Parameter[float]
    g_cr_ts: Parameter[float]
    tk_vv_single_wall: Parameter[float]
    ep_width: Parameter[float]
    ep_height: Parameter[float]
    ep_z_position: Parameter[float]


class TSEquatorialPortDuctBuilder(Builder):
    """Thermal shield upper port duct builder"""

    params: TSEquatorialPortDuctBuilderParams
    param_cls = TSEquatorialPortDuctBuilderParams

    TS = "TS"

    def __init__(
        self,
        params: dict | ParameterFrame | ConfigParams | None,
        build_config: dict,
        cryostat_xz: BluemiraWire,
    ):
        super().__init__(params, build_config)
        # Put the end of the equatorial port half-way between cryostat ts and
        # cryostat
        self.x_max = cryostat_xz.bounding_box.x_max + 0.5 * self.params.g_cr_ts.value

    def build(self) -> Component:
        """
        Build equatorial port

        Returns
        -------
        :
            The component tree
        """
        offset = (
            self.params.tk_vv_single_wall.value
            + self.params.g_vv_ts.value
            + self.params.tk_ts.value
        )
        y_val = 0.5 * self.params.ep_width.value + offset
        z_ref = self.params.ep_z_position.value
        z_val = 0.5 * self.params.ep_height.value + offset
        yz_face = make_equatorial_port_yz_face(
            self.x_max,
            -y_val,
            y_val,
            z_ref - z_val,
            z_ref + z_val,
            self.params.tk_ts.value,
        )

        return self.component_tree(None, None, self.build_xyz(yz_face))

    def build_xyz(self, yz_face: BluemiraFace) -> list[PhysicalComponent]:
        """
        Build equatorial port xyz

        Returns
        -------
        :
            The xyz components
        """
        yz_voidface = BluemiraFace(yz_face.boundary[1])
        degree = 180 / self.params.n_TF.value
        vec = (self.params.R_0.value - self.x_max, 0, 0)
        port = extrude_shape(yz_face, vec)
        port.rotate(degree=degree)
        comp = PhysicalComponent(
            self.name,
            port,
            self.get_material(self.TS),
        )

        void = extrude_shape(yz_voidface, vec)
        void.rotate(degree=degree)
        void = PhysicalComponent(
            self.name + " voidspace", void, material=Void("vacuum")
        )

        apply_component_display_options(comp, BLUE_PALETTE[self.TS][0])
        apply_component_display_options(void, color=(0, 0, 0))
        return [comp, void]


@dataclass
class VVUpperPortDuctBuilderParams(ParameterFrame):
    """Vacuum vessel upper port duct builder Parameter Frame"""

    n_TF: Parameter[int]
    tf_wp_depth: Parameter[float]
    g_ts_tf: Parameter[float]
    tk_ts: Parameter[float]
    g_vv_ts: Parameter[float]
    g_cr_ts: Parameter[float]
    tk_vv_double_wall: Parameter[float]
    tk_vv_single_wall: Parameter[float]
    upper_port_expanded_section_depth_from_top_pct: Parameter[float]
    upper_port_expand_radial_outward_pct: Parameter[float]
    upper_port_expand_radial_inward_pct: Parameter[float]
    upper_port_expand_lateral_pct: Parameter[float]


class VVUpperPortDuctBuilder(Builder):
    """Vacuum vessel upper port duct builder"""

    params: VVUpperPortDuctBuilderParams
    param_cls = VVUpperPortDuctBuilderParams

    VV = "VV"

    def __init__(
        self,
        params: dict | ParameterFrame | ConfigParams | None,
        build_config: dict,
        port_koz: BluemiraFace,
        cryostat_ts_xz: BluemiraWire,
    ):
        super().__init__(params, build_config)
        koz_offset = self.params.tk_ts.value + self.params.g_vv_ts.value
        x_min_ts_base, x_max_ts_base = _base_x_bounds_from_koz_bbox(
            port_koz.bounding_box.x_min,
            port_koz.bounding_box.x_max,
            self.params.upper_port_expand_radial_inward_pct.value,
            self.params.upper_port_expand_radial_outward_pct.value,
        )
        self.x_min = x_min_ts_base + koz_offset
        self.x_max = x_max_ts_base - koz_offset
        self.z_max = cryostat_ts_xz.bounding_box.z_max + 0.5 * self.params.g_cr_ts.value

        if (
            self.params.tk_vv_double_wall.value <= 0
            or self.params.tk_vv_single_wall.value <= 0
        ):
            raise ValueError("Port wall thickness must be > 0")

        self.y_offset = (
            self.params.tf_wp_depth.value
            + self.params.g_ts_tf.value
            + self.params.tk_ts.value
            + self.params.g_vv_ts.value
        )

    def build(self) -> Component:
        """
        Build upper port

        Returns
        -------
        :
            The upper port component tree
        """
        xy_face = make_upper_port_xy_face(
            self.params.n_TF.value,
            self.x_min,
            self.x_max,
            self.params.tk_vv_double_wall.value,
            self.params.tk_vv_single_wall.value,
            self.y_offset,
        )

        return self.component_tree(
            None,
            self.build_xy(xy_face),
            self.build_xyz(xy_face),
        )

    def build_xyz(self, xy_face: BluemiraFace) -> list[PhysicalComponent]:
        """
        Build upper port xyz

        Returns
        -------
        :
            The xyz components
        """
        expand_any = any(
            abs(v) > GEOM_TOL
            for v in (
                self.params.upper_port_expanded_section_depth_from_top_pct.value,
                self.params.upper_port_expand_radial_outward_pct.value,
                self.params.upper_port_expand_radial_inward_pct.value,
                self.params.upper_port_expand_lateral_pct.value,
            )
        )

        if not expand_any:
            xy_voidface = BluemiraFace(xy_face.boundary[1])
            xy_outface = BluemiraFace(xy_face.boundary[0])
            port = extrude_shape(xy_face, (0, 0, self.z_max))
            # Add start-cap for future boolean fragmentation help
            cap = extrude_shape(xy_outface, vec=(0, 0, 0.1))
            port = boolean_fuse([port, cap])

            comp = PhysicalComponent(
                self.name,
                port,
                material=self.get_material(self.VV),
            )
            apply_component_display_options(comp, BLUE_PALETTE[self.VV][0])
            void = PhysicalComponent(
                self.name + " voidspace",
                extrude_shape(xy_voidface, (0, 0, self.z_max)),
                material=Void("vacuum"),
            )
            apply_component_display_options(void, color=(0, 0, 0))
            return [comp, void]

        y_expand = _side_expand_m(
            self.x_min,
            self.x_max,
            self.params.upper_port_expand_lateral_pct.value,
        )
        x_min_outer, x_max_outer = _expanded_x_bounds(
            self.x_min,
            self.x_max,
            self.params.upper_port_expand_radial_inward_pct.value,
            self.params.upper_port_expand_radial_outward_pct.value,
        )
        wall_end_tk = self.params.tk_vv_double_wall.value
        wall_side_tk = self.params.tk_vv_single_wall.value
        expand_from_top_m = _pct_to_m(
            self.params.upper_port_expanded_section_depth_from_top_pct.value, self.z_max
        )

        xy_base = xy_face
        xy_shelf = _make_upper_port_xy_face_with_shelf(
            self.params.n_TF.value,
            x_min_base=self.x_min,
            x_max_base=self.x_max,
            x_min_outer=x_min_outer,
            x_max_outer=x_max_outer,
            wall_end_tk=wall_end_tk,
            wall_side_tk=wall_side_tk,
            y_offset=self.y_offset,
            y_expand=y_expand,
        )
        if abs(y_expand) > GEOM_TOL:
            xy_top = _make_upper_port_xy_face_y_expanded(
                self.params.n_TF.value,
                x_min_outer,
                x_max_outer,
                wall_end_tk,
                wall_side_tk,
                self.y_offset,
                y_expand,
                allow_outer_trim=False,
                allow_inner_trim=False,
            )
        else:
            xy_top = _make_upper_port_xy_face_no_outer_trim(
                self.params.n_TF.value,
                x_min_outer,
                x_max_outer,
                wall_end_tk,
                wall_side_tk,
                self.y_offset,
            )

        port = _extrude_three_parts(
            xy_base,
            xy_shelf,
            xy_top,
            self.z_max,
            shelf_h=max(wall_end_tk, wall_side_tk),
            expand_from_top_m=expand_from_top_m,
        )

        xy_outface = BluemiraFace(xy_base.boundary[0])
        cap = extrude_shape(xy_outface, vec=(0, 0, 0.1))
        port = boolean_fuse([port, cap])

        void = _extrude_three_parts(
            BluemiraFace(xy_base.boundary[1]),
            BluemiraFace(xy_shelf.boundary[1]),
            BluemiraFace(xy_top.boundary[1]),
            self.z_max,
            shelf_h=max(wall_end_tk, wall_side_tk),
            expand_from_top_m=expand_from_top_m,
        )

        comp = PhysicalComponent(self.name, port, material=self.get_material(self.VV))
        void = PhysicalComponent(
            self.name + " voidspace", void, material=Void("vacuum")
        )
        apply_component_display_options(comp, BLUE_PALETTE[self.VV][0])
        apply_component_display_options(void, color=(0, 0, 0))
        return [comp, void]

    def build_xy(self, xy_face: BluemiraFace) -> list[PhysicalComponent]:
        """
        Build upper port xy face

        Returns
        -------
        :
            The xy components
        """
        xy_voidface = BluemiraFace(xy_face.boundary[1])
        comp = PhysicalComponent(self.name, xy_face)
        apply_component_display_options(comp, BLUE_PALETTE[self.VV][0])
        void = PhysicalComponent(
            self.name + " voidspace", xy_voidface, material=Void("vacuum")
        )
        apply_component_display_options(void, color=(0, 0, 0))
        return [comp, void]


@dataclass
class VVEquatorialPortDuctBuilderParams(ParameterFrame):
    """Vacuum vessel equatorial port duct builder Parameter Frame"""

    R_0: Parameter[float]
    n_TF: Parameter[int]
    g_cr_ts: Parameter[float]
    ep_z_position: Parameter[float]
    ep_width: Parameter[float]
    ep_height: Parameter[float]
    tk_vv_single_wall: Parameter[float]


class VVEquatorialPortDuctBuilder(Builder):
    """Vacuum vessel upper port duct builder"""

    params: VVEquatorialPortDuctBuilderParams
    param_cls = VVEquatorialPortDuctBuilderParams

    VV = "VV"

    def __init__(
        self,
        params: dict | ParameterFrame | ConfigParams | None,
        build_config: dict,
        cryostat_xz: BluemiraWire,
    ):
        super().__init__(params, build_config)
        # Put the end of the equatorial port half-way between cryostat ts and
        # cryostat
        self.x_max = cryostat_xz.bounding_box.x_max + 0.5 * self.params.g_cr_ts.value

    def build(self) -> Component:
        """
        Build equatorial port

        Returns
        -------
        :
            The component tree
        """
        y_val = 0.5 * self.params.ep_width.value
        z_ref = self.params.ep_z_position.value
        z_val = 0.5 * self.params.ep_height.value
        yz_face = make_equatorial_port_yz_face(
            self.x_max,
            -y_val,
            y_val,
            z_ref - z_val,
            z_ref + z_val,
            self.params.tk_vv_single_wall.value,
        )

        return self.component_tree(
            None,
            None,
            self.build_xyz(yz_face),
        )

    def build_xyz(self, yz_face: BluemiraFace) -> PhysicalComponent:
        """
        Build equatorial port xyz

        Returns
        -------
        :
            The xyz shapes of the equatorial port duct
        """
        yz_voidface = BluemiraFace(yz_face.boundary[1])
        degree = 180 / self.params.n_TF.value
        vec = (self.params.R_0.value - self.x_max, 0, 0)
        port = extrude_shape(yz_face, vec)
        port.rotate(degree=degree)
        comp = PhysicalComponent(
            self.name,
            port,
            material=self.get_material(self.VV),
        )

        void = extrude_shape(yz_voidface, vec)
        void.rotate(degree=degree)
        void = PhysicalComponent(
            self.name + " voidspace", void, material=Void("vacuum")
        )

        apply_component_display_options(comp, BLUE_PALETTE[self.VV][0])
        apply_component_display_options(void, color=(0, 0, 0))
        return [comp, void]


def _pct_to_m(pct: float, base: float) -> float:
    return 0.01 * pct * base


def _expanded_x_bounds(
    x_min: float,
    x_max: float,
    radial_inward_pct: float,
    radial_outward_pct: float,
) -> tuple[float, float]:
    x_span = x_max - x_min
    if x_span <= 0:
        raise BuilderError("Port dimensions too small")
    return (
        x_min - _pct_to_m(radial_inward_pct, x_span),
        x_max + _pct_to_m(radial_outward_pct, x_span),
    )


def _side_expand_m(x_min: float, x_max: float, lateral_pct: float) -> float:
    x_span = x_max - x_min
    if x_span <= 0:
        raise BuilderError("Port dimensions too small")
    return _pct_to_m(lateral_pct, x_span)


def _base_x_bounds_from_koz_bbox(
    x_min_bbox: float,
    x_max_bbox: float,
    radial_inward_pct: float,
    radial_outward_pct: float,
) -> tuple[float, float]:
    """Recover unexpanded base X bounds from a KOZ bbox."""
    x_span_bbox = x_max_bbox - x_min_bbox
    if x_span_bbox <= 0:
        raise BuilderError("Port dimensions too small")

    if (
        abs(radial_inward_pct) <= GEOM_TOL
        and abs(radial_outward_pct) <= GEOM_TOL
    ):
        return x_min_bbox, x_max_bbox

    scale = 1 + 0.01 * (radial_inward_pct + radial_outward_pct)
    if scale <= GEOM_TOL:
        raise BuilderError("Invalid radial expansion percentages")

    x_span_base = x_span_bbox / scale
    x_min_base = x_min_bbox + _pct_to_m(radial_inward_pct, x_span_base)
    x_max_base = x_max_bbox - _pct_to_m(radial_outward_pct, x_span_base)
    return x_min_base, x_max_base


def _make_upper_port_xy_face_y_expanded(
    n_TF: int,
    x_min: float,
    x_max: float,
    wall_end_tk: float,
    wall_side_tk: float,
    y_offset: float,
    y_expand: float,
    *,
    allow_outer_trim: bool = True,
    allow_inner_trim: bool = True,
) -> BluemiraFace:
    half_beta = np.pi / n_TF
    cos_hb = np.cos(half_beta)
    tan_hb = np.tan(half_beta)

    y_tf_out = y_offset / cos_hb
    y_tf_in = y_tf_out + wall_side_tk / cos_hb

    x1 = x_min

    a1 = 1 + tan_hb**2
    b1 = -2 * y_tf_out * tan_hb
    c1 = y_tf_out**2 - x_max**2
    discriminant = b1**2 - 4 * a1 * c1
    x4 = 0.5 * (-b1 + np.sqrt(discriminant)) / a1

    x2, x3 = x1 + wall_end_tk, x4 - wall_end_tk

    if x2 >= x3:
        raise BuilderError("Port dimensions too small")

    y1 = x1 * tan_hb - y_tf_out

    if y1 < 0 and allow_outer_trim:
        y1 = 0
        x1 = y_tf_out / tan_hb
        x2 = x1 + wall_end_tk

    y2, y3 = x2 * tan_hb - y_tf_in, x3 * tan_hb - y_tf_in

    if y3 <= 0:
        raise BuilderError("Port dimensions too small")

    if y2 < 0 and allow_inner_trim:
        y2 = 0
        c = y3 - tan_hb * x3
        x2 = -c / tan_hb
        x1 = x2 - wall_end_tk

    y1, y4 = x1 * tan_hb - y_tf_out, x4 * tan_hb - y_tf_out

    if abs(y_expand) > GEOM_TOL:
        y1 += y_expand
        y2 += y_expand
        y3 += y_expand
        y4 += y_expand

    inner_wire = make_polygon(
        {"x": [x2, x3, x3, x2], "y": [-y2, -y3, y3, y2]}, closed=True
    )
    outer_wire = make_polygon(
        {"x": [x1, x4, x4, x1], "y": [-y1, -y4, y4, y1]}, closed=True
    )

    xy_face = BluemiraFace((outer_wire, inner_wire))
    xy_face.rotate(degree=np.rad2deg(half_beta))

    return xy_face


def _make_upper_port_xy_face_no_outer_trim(
    n_TF: int,
    x_min: float,
    x_max: float,
    wall_end_tk: float,
    wall_side_tk: float,
    y_offset: float,
) -> BluemiraFace:
    return _make_upper_port_xy_face_y_expanded(
        n_TF,
        x_min,
        x_max,
        wall_end_tk,
        wall_side_tk,
        y_offset,
        y_expand=0.0,
        allow_outer_trim=False,
        allow_inner_trim=False,
    )


def _make_upper_port_xy_face_with_shelf(
    n_tf: int,
    x_min_base: float,
    x_max_base: float,
    x_min_outer: float,
    x_max_outer: float,
    wall_end_tk: float,
    wall_side_tk: float,
    y_offset: float,
    y_expand: float,
) -> BluemiraFace:
    if abs(y_expand) > GEOM_TOL:
        outer_face = _make_upper_port_xy_face_y_expanded(
            n_tf,
            x_min_outer,
            x_max_outer,
            wall_end_tk,
            wall_side_tk,
            y_offset,
            y_expand,
            allow_outer_trim=False,
            allow_inner_trim=False,
        )
    else:
        outer_face = _make_upper_port_xy_face_no_outer_trim(
            n_tf,
            x_min_outer,
            x_max_outer,
            wall_end_tk,
            wall_side_tk,
            y_offset,
        )

    inner_face = make_upper_port_xy_face(
        n_tf, x_min_base, x_max_base, wall_end_tk, wall_side_tk, y_offset
    )

    return BluemiraFace((outer_face.boundary[0], *inner_face.boundary[1:]))


def _extrude_segment(face: BluemiraFace, z0: float, height: float):
    if height <= GEOM_TOL:
        return None
    solid = extrude_shape(face, (0, 0, height))
    if z0 > GEOM_TOL:
        solid.translate((0, 0, z0))
    return solid


def _fuse_nonempty(shapes):
    shapes = [s for s in shapes if s is not None]
    if not shapes:
        return None
    if len(shapes) == 1:
        return shapes[0]
    return boolean_fuse(shapes)


def _extrude_three_parts(
    xy_base: BluemiraFace,
    xy_shelf: BluemiraFace,
    xy_top: BluemiraFace,
    z_max: float,
    shelf_h: float,
    expand_from_top_m: float,
):
    z_cut = max(0.0, z_max - expand_from_top_m)
    z2 = min(z_max, z_cut + max(0.0, shelf_h))

    base_solid = _extrude_segment(xy_base, z0=0.0, height=max(0.0, z_cut))
    shelf_solid = _extrude_segment(xy_shelf, z0=z_cut, height=max(0.0, z2 - z_cut))
    top_solid = _extrude_segment(xy_top, z0=z2, height=max(0.0, z_max - z2))
    return _fuse_nonempty([base_solid, shelf_solid, top_solid])


def make_upper_port_xy_face(
    n_TF: int,
    x_min: float,
    x_max: float,
    wall_end_tk: float,
    wall_side_tk: float,
    y_offset: float,
) -> BluemiraFace:
    """
    Creates a xy cross section of the upper port

    translates the port koz to the origin,
    builds the port at the origin and moves it back

    Parameters
    ----------
    n_TF:
        Number of TF coils
    x_min:
        Inner radius of the port keep-out zone
    x_max:
        Outer radius of the port keep-out zone
    wall_end_tk:
        Port wall end thickness
    wall_size_tk:
        Port wall side thickness
    y_offset:
        Offset value from the x-z plane at which to start building the port
        (excluding port side wall thickness)

    Returns
    -------
    xy_face:
        x-y face of the upper port

    Raises
    ------
    BuilderError
        Port dimensions too small to construct

    Notes
    -----
    the port koz is slightly trimmed to allow for square ends to the port
    """
    half_beta = np.pi / n_TF
    cos_hb = np.cos(half_beta)
    tan_hb = np.tan(half_beta)

    y_tf_out = y_offset / cos_hb
    y_tf_in = y_tf_out + wall_side_tk / cos_hb

    x1 = x_min

    # This is the correct way to retrieve the outer radius of the port, accounting
    # for the TF thickness
    # It's just the intersection of a line and a circle in the positive quadrant.
    a1 = 1 + tan_hb**2
    b1 = -2 * y_tf_out * tan_hb
    c1 = y_tf_out**2 - x_max**2
    discriminant = b1**2 - 4 * a1 * c1
    x4 = 0.5 * (-b1 + np.sqrt(discriminant)) / a1

    x2, x3 = x1 + wall_end_tk, x4 - wall_end_tk

    if x2 >= x3:
        raise BuilderError("Port dimensions too small")

    y1 = x1 * tan_hb - y_tf_out

    if y1 < 0:
        # Triangular outer port wall
        y1 = 0
        x1 = y_tf_out / tan_hb
        x2 = x1 + wall_end_tk

    y2, y3 = x2 * tan_hb - y_tf_in, x3 * tan_hb - y_tf_in

    if y3 <= 0:
        raise BuilderError("Port dimensions too small")

    if y2 < 0:
        # Triangular inner port wall
        y2 = 0
        c = y3 - tan_hb * x3
        x2 = -c / tan_hb
        x1 = x2 - wall_end_tk

    y1, y4 = x1 * tan_hb - y_tf_out, x4 * tan_hb - y_tf_out

    inner_wire = make_polygon(
        {"x": [x2, x3, x3, x2], "y": [-y2, -y3, y3, y2]}, closed=True
    )
    outer_wire = make_polygon(
        {"x": [x1, x4, x4, x1], "y": [-y1, -y4, y4, y1]}, closed=True
    )

    xy_face = BluemiraFace((outer_wire, inner_wire))
    xy_face.rotate(degree=np.rad2deg(half_beta))

    return xy_face


def make_equatorial_port_yz_face(
    x_ref: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    wall_side_tk: float,
) -> BluemiraFace:
    """
    Creates a yz cross section of the equatorial port

    builds the port at the origin

    Parameters
    ----------
    x_ref:
        Reference x coordinate of the y-z plane
    y_min:
        Minimum y coordinate of the port void
    y_max:
        Maximum y coordinate of the port void
    z_min:
        Minimum z coordinate of the port void
    z_max:
        Maximum z coordinate of the port void
    wall_side_tk:
        Thickness of the port walss

    Returns
    -------
    yz_face:
        y-z face of the equatorial port
    """
    y = np.array([y_min, y_min, y_max, y_max])
    z = np.array([z_min, z_max, z_max, z_min])
    inner = make_polygon({"x": x_ref, "y": y, "z": z}, closed=True)
    outer = offset_wire(inner, wall_side_tk, open_wire=False)
    return BluemiraFace([outer, inner])


def pipe_pipe_join(
    target_shape: BluemiraSolid,
    target_void: BluemiraSolid,
    tool_shape: BluemiraSolid,
    tool_void: BluemiraSolid,
) -> list[BluemiraSolid]:
    """
    Join two hollow, intersecting pipes.

    Parameters
    ----------
    target_shape:
        Solid of the target shape
    target_void:
        Solid of the target void
    tool_shape:
        Solid of the tool shape
    tool_void:
        Solid of the tool void

    Returns
    -------
    shape:
        Solid of the joined pipe-pipe shape
    void:
        Solid of the joined pipe-pipe void

    Notes
    -----
    This approach is more brittle than a classic fuse, fuse, cut operation, but is
    substantially faster. If the parts do not fully intersect, undesired results
    are to be expected.
    """
    _, (target_fragments, tool_fragments) = boolean_fragments(
        [target_shape, tool_shape]
    )

    # Keep the largest piece of the target by volume (opinionated)
    # This is in case its COG is inside the tool void
    target_fragments.sort(key=lambda solid: -solid.volume)
    new_shape_pieces = [target_fragments[0]]
    target_fragments = target_fragments[1:]
    for targ_frag in target_fragments:
        # Find the target piece(s) that are inside the tool
        com = targ_frag.center_of_mass
        if not point_inside_shape(com, tool_void):
            new_shape_pieces.append(targ_frag)

    for tool_frag in tool_fragments:
        # Find the tool piece(s) that are inside the target
        com = tool_frag.center_of_mass
        if not point_inside_shape(com, target_void):
            new_shape_pieces.append(tool_frag)
        else:
            # Find the union piece(s)
            new_shape_pieces.extend(
                [
                    tool_frag
                    for targ_frag in target_fragments
                    if tool_frag.is_same(targ_frag)
                ]
            )

    return new_shape_pieces
