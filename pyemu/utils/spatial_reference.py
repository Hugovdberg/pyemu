# pyright: strict
import warnings
from collections import defaultdict
from typing import Any, Dict, Protocol, Tuple, TypeVar, Union, runtime_checkable

import numpy as np
from pyemu.logger import SuperLogger

from ..pyemu_warnings import AmbiguousIndex

T = TypeVar("T")
KT = TypeVar("KT", contravariant=True)
VT = TypeVar("VT", covariant=True)
SRDict = Dict[Union[int, Tuple[int, ...]], Tuple[float, ...]]


@runtime_checkable
class GetItem(Protocol[KT, VT]):
    def __getitem__(self, *keys: KT) -> VT:
        ...


@runtime_checkable
class FlopySR(Protocol):
    delr: np.ndarray[Tuple[int], np.dtype[np.float64]]
    delc: np.ndarray[Tuple[int], np.dtype[np.float64]]
    xcentergrid: np.ndarray[Tuple[int, int], np.dtype[np.float64]]
    ycentergrid: np.ndarray[Tuple[int, int], np.dtype[np.float64]]


@runtime_checkable
class FlopyMG(Protocol):
    delr: np.ndarray[Tuple[int], np.dtype[np.float64]]
    delc: np.ndarray[Tuple[int], np.dtype[np.float64]]
    xcellcenters: np.ndarray[Tuple[int, int], np.dtype[np.float64]]
    ycellcenters: np.ndarray[Tuple[int, int], np.dtype[np.float64]]


class SpatialReference:
    def __init__(
        self,
        spatial_reference: Any,
        parent_logger: SuperLogger,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._spatial_reference = spatial_reference
        self.logger = parent_logger.getChild(self.__class__.__name__)
        self.ijwarned: Dict[int, bool] = defaultdict(bool)
        self.add_pars_callcount: int = 0

    def get_xy(
        self, args: Tuple[int, ...], **kwargs: Any
    ) -> Union[Tuple[int, int], Tuple[float, float], Tuple[None, None]]:
        return self._parse_kij_args(args, kwargs)

    def is_regular_grid(self, tolerance: float = 1e-4) -> bool:
        """Flag indicating whether the SpatialReference refers to a regular spaced grid.

        Args:
            tolerance (float, optional): Relative tolerance to determine grid regularity. Defaults to 1e-4.

        Returns:
            bool: flag indicating if the grid is regular.
        """
        return False

    def _parse_kij_args(
        self, args: Tuple[int, ...], kwargs: Dict[str, Any]
    ) -> Union[Tuple[int, int], Tuple[None, None]]:
        """parse args into kij indices."""
        if len(args) >= 2:
            ij_id = None
            if "ij_id" in kwargs:
                ij_id = kwargs["ij_id"]
            if ij_id is not None:
                i, j = [args[ij] for ij in ij_id]
            else:
                warnings.warn(
                    "Position of i and j in index_cols not specified, "
                    "assume (i,j) are final two entries in index_cols.",
                    category=AmbiguousIndex,
                    stacklevel=3,
                )
                i, j = args[-2], args[-1]
            return (i, j)
        else:
            warnings.warn(
                "get_xy() warning: need locational information "
                "(e.g. i,j) to generate xy, "
                f"insufficient index cols passed to interpret: {args!s}",
                category=AmbiguousIndex,
                stacklevel=3,
            )
            return (None, None)


class DictSpatialReference(SpatialReference):
    def __init__(
        self,
        spatial_reference: SRDict,
        parent_logger: SuperLogger,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(spatial_reference, parent_logger, *args, **kwargs)
        self._spatial_reference: SRDict = spatial_reference

    def get_xy(
        self, args: Tuple[int, ...], **kwargs: Any
    ) -> Union[Tuple[float, float], Tuple[None, None]]:
        if isinstance(args, list):
            args = tuple(args)
        xy = self._spatial_reference.get(args, None)
        base_error_msg = f"error getting xy from arg:'{args}' - {{err}}"
        if xy is None:
            arg_len = None
            try:
                arg_len = len(args)
            except TypeError as e:
                msg = base_error_msg.format(err="no len support")
                raise self.logger.logged_exception(msg, ValueError) from e
            if arg_len == 1:
                xy = self._spatial_reference.get(args[0], None)
            elif arg_len == 2 and args[0] == 0:
                xy = self._spatial_reference.get(args[1], None)
            elif arg_len == 2 and args[1] == 0:
                xy = self._spatial_reference.get(args[0], None)
            else:
                raise self.logger.logged_exception(
                    base_error_msg.format(err="no value found"), ValueError
                )
        if xy is None:
            raise self.logger.logged_exception(
                base_error_msg.format(err="still None..."), ValueError
            )
        return xy[0], xy[1]


class FlopySRSpatialReference(SpatialReference):
    def __init__(
        self,
        spatial_reference: FlopySR,
        parent_logger: SuperLogger,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(spatial_reference, parent_logger, *args, **kwargs)
        self._spatial_reference: FlopySR = spatial_reference

    def get_xy(
        self, args: Tuple[int, ...], **kwargs: Any
    ) -> Union[Tuple[float, float], Tuple[None, None]]:
        ij = self._parse_kij_args(args, kwargs)
        if ij[0] is None:
            return (None, None)
        else:
            i, j = ij
            return (
                self._spatial_reference.xcentergrid[i, j],
                self._spatial_reference.ycentergrid[i, j],
            )

    def is_regular_grid(self, tolerance: float = 1e-4) -> bool:
        delx = self._spatial_reference.delc
        dely = self._spatial_reference.delr
        if np.abs((delx.mean() - delx.min()) / delx.mean()) > tolerance:  # type: ignore
            return False
        if (np.abs(dely.mean() - dely.min()) / dely.mean()) > tolerance:  # type: ignore
            return False
        if (np.abs(delx.mean() - dely.mean()) / delx.mean()) > tolerance:  # type: ignore
            return False
        return True


class FlopyMGSpatialReference(SpatialReference):
    def __init__(
        self,
        spatial_reference: FlopyMG,
        parent_logger: SuperLogger,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(spatial_reference, parent_logger, *args, **kwargs)
        self._spatial_reference: FlopyMG = spatial_reference

    def get_xy(
        self, args: Tuple[int, ...], **kwargs: Any
    ) -> Union[Tuple[float, float], Tuple[None, None]]:
        ij = self._parse_kij_args(args, kwargs)
        if ij[0] is None:
            return (None, None)
        else:
            i, j = ij
            return (
                self._spatial_reference.xcellcenters[i, j],
                self._spatial_reference.ycellcenters[i, j],
            )

    def is_regular_grid(self, tolerance: float = 1e-4) -> bool:
        delx = self._spatial_reference.delc
        dely = self._spatial_reference.delr
        if np.abs((delx.mean() - delx.min()) / delx.mean()) > tolerance:  # type: ignore
            return False
        if (np.abs(dely.mean() - dely.min()) / dely.mean()) > tolerance:  # type: ignore
            return False
        if (np.abs(delx.mean() - dely.mean()) / delx.mean()) > tolerance:  # type: ignore
            return False
        return True


def initialize_spatial_reference(
    spatial_reference: Union[None, FlopySR, FlopyMG, SRDict, SpatialReference],
    parent_logger: SuperLogger,
) -> SpatialReference:
    """process the spatial reference argument.  Called programmatically"""
    if spatial_reference is None:
        return SpatialReference(spatial_reference, parent_logger)
    elif isinstance(spatial_reference, FlopySR):
        return FlopySRSpatialReference(spatial_reference, parent_logger)
    elif isinstance(spatial_reference, FlopyMG):
        return FlopyMGSpatialReference(spatial_reference, parent_logger)
    elif isinstance(spatial_reference, dict):
        parent_logger.info("dictionary-based spatial reference detected...")
        return DictSpatialReference(spatial_reference, parent_logger)
    elif isinstance(spatial_reference, SpatialReference):  # type: ignore
        return spatial_reference
    else:
        raise parent_logger.logged_exception(
            "initialize_spatial_reference() error: unsupported spatial_reference",
            TypeError,
        )
