# Re-written from tsai (https://github.com/timeseriesAI/tsai) for determinism

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator, interp1d

__all__ = ['AddNoise', 'Drift', 'Dropout', 'TimeWarp']

class AddNoise:
    '''Add random noise to time series. The noise added to every time point of
    a time series is independent and identically distributed.'''

    def __init__(
        self,
        loc: float | tuple[float, float] | list[float] = 0,
        scale: float | tuple[float, float] | list[float] = 0.1,
        distr: str = 'gaussian',
        kind: str = 'additive',
        per_channel: bool = True,
    ) -> None:
        '''Add random noise to time series. The noise added to every time
        point of a time series is independent and identically distributed.

        Args:
            loc (float | tuple[float, float] | list[float], optional): Mean of
            the ranodm noise. If float, all noise value are sampled with the
            same mean. If tuple, the noise added to a series (a channel if
            `per_channel` is True) is sampled from a distribution with a
            standard deviation that is randomly selected from the interval.
            If list, the noise added to a series (a channel if `per_channel`
            is True) is sampled from a distribution with a mean that is
            randomly selected from the list. Defaults to 0.
            scale (float | tuple[float, float] | list[float], optional):
            Standard deviation of the random noise. If float, all noise value
            are sampled with the same standard deviation. If tuple, the noise
            added to a series (a channel if `per_channel` is True) is sampled
            from a distribution with a standard deviation that is randomly
            selected from the interval. If list, the noise added to a series
            (a channel if `per_channel` is True) is sampled from a
            distribution with a standard deviation that is randomly selected
            from the list. Defaults to 0.1.
            distr (str, optional): Distribution of the random noise. It must
            be one of "gaussian", "laplace", and "uniform". Defaults to
            "gaussian".
            kind (str, optional): How the noise is added to the original time
            series. It must be either "additive" or "multiplicative". Defaults
            to "additive".
            per_channel (bool, optional): Whether to sample independent noise
            values for each channel in a time series or to use the same noise
            for all channels in a time series. Defaults to True.
        '''
        self.loc = loc
        self.scale = scale
        self.distr = distr
        self.kind = kind
        self.per_channel = per_channel

        # set up noise generator
        if distr == 'gaussian':
            self.gen_noise = lambda size: np.random.normal(0.0, 1.0, size)
        elif distr == 'laplace':
            self.gen_noise = lambda size: np.random.laplace(0.0, 1.0, size)
        elif distr == 'uniform':
            self.gen_noise = lambda size: np.random.uniform(0.0, 1.0, size)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        '''`__call__` method.

        Args:
            x (numpy.ndarray): Input sequence to process.

        Returns:
            numpy.ndarray: Processed output sequence.
        '''
        L, C = x.shape

        if isinstance(self.loc, (float, int)):
            loc = self.loc
        elif isinstance(self.loc, tuple):
            loc = np.random.uniform(low=self.loc[0], high=self.loc[1])
        elif isinstance(self.loc, list):
            loc = np.random.choice(self.loc)

        if isinstance(self.scale, (float, int)):
            scale = self.scale
        elif isinstance(self.scale, tuple):
            scale = np.random.uniform(low=self.scale[0], high=self.scale[1])
        elif isinstance(self.scale, list):
            scale = np.random.choice(self.scale)

        if self.per_channel:
            noise = self.gen_noise((L, C))
        else:
            noise = self.gen_noise((L, 1))
            noise = np.repeat(noise, C, axis=1)

        noise = noise * scale + loc

        if self.kind == 'additive':
            x = x + noise
        else:
            x = x * (1.0 + noise)

        return x


class Drift:
    '''Drift the value of time series. The augmenter drifts the value of time
    series from its original values randomly and smoothly. The extent of
    drifting is controlled by the maximal drift and the number of drift points.
    '''

    def __init__(
        self,
        max_drift: float | tuple[float, float] = 0.5,
        n_drift_points: int | list[int] = 3,
        kind: str = 'additive',
        per_channel: bool = True,
    ) -> None:
        '''Drift the value of time series. The augmenter drifts the value of
        time series from its original values randomly and smoothly. The extent
        of drifting is controlled by the maximal drift and the number of drift
        points.

        Args:
            max_drift (float | tuple[float, float], optional): The maximal
            amount of drift added to a time series. If float, all series (all
            channels if `per_channel` is True) are drifted with the same
            maximum. If tuple, the maximal drift added to a time series (a
            channel if `per_channel` is True) is sampled from this interval
            randomly. Defaults to 0.5.
            n_drift_points (int | list[int], optional): The number of time
            points a new drifting trend is defined in a series. If int, all
            series (all channels if `per_channel` is True) have the same
            number of drift points. If list, the number of drift points
            defined in a series (a channel if `per_channel` is True) is
            sampled from this list randomly. Defaults to 3.
            kind (str, optional): How the noise is added to the original time
            series. It must be either "additive" or "multiplicative". Defaults
            to "additive".
            per_channel (bool, optional): Whether to sample independent
            drifting trends for each channel in a time series or to use the
            same drifting trends for all channels in a time series. Defaults
            to True.
        '''
        self.max_drift = max_drift
        self.kind = kind
        self.per_channel = per_channel

        if isinstance(n_drift_points, int):
            self.n_drift_points = set([n_drift_points])
        else:
            self.n_drift_points = set(n_drift_points)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        '''`__call__` method.

        Args:
            x (numpy.ndarray): Input sequence to process.

        Returns:
            numpy.ndarray: Processed output sequence.
        '''
        L, C = x.shape
        ind = np.random.choice(
            len(self.n_drift_points), C if self.per_channel else 1
        )  # map seriese to n_drift_points
        drift = np.zeros((C if self.per_channel else 1, L))

        for i, n in enumerate(self.n_drift_points):
            if not (ind == i).any():
                continue

            anchors = np.cumsum(
                np.random.normal(size=((ind == i).sum(), n + 2)), axis=1
            )
            interpFuncs = CubicSpline(
                np.linspace(0, L, n + 2), anchors, axis=1
            )
            drift[ind == i, :] = interpFuncs(np.arange(L))

        drift = drift.reshape((-1, L)).swapaxes(0, 1)
        drift = drift - drift[0, :].reshape(1, -1)
        drift = drift / (abs(drift).max(axis=1, keepdims=True) + 1e-6)

        if isinstance(self.max_drift, (float, int)):
            drift = drift * self.max_drift
        else:
            drift = drift * np.random.uniform(
                low=self.max_drift[0],
                high=self.max_drift[1],
                size=(1, C if self.per_channel else 1),
            )

        if self.kind == 'additive':
            x = x + drift
        else:
            x = x * (1 + drift)

        return x


class Dropout:
    '''Dropout values of some random time points in time series. Single time
    points or sub-sequences could be dropped out.'''

    def __init__(
        self,
        p: float | tuple[float, float] | list[float] = 0.05,
        size: int | tuple[int, int] | list[int] = 1,
        fill: str | float = 'ffill',
        per_channel: bool = False,
    ) -> None:
        '''Dropout values of some random time points in time series. Single
        time points or sub-sequences could be dropped out.

        Args:
            p (float | tuple[float, float] | list[float], optional):
            Probablity of the value of a time point to be dropped out. If
            float, all series (all channels if `per_channel` is True) have the
            same probability. If tuple, a series (a channel if `per_channel`
            is True) has a probability sampled from this interval randomly. If
            list, a series (a channel if `per_channel` is True) has a
            probability sampled from this list randomly. Defaults to 0.05.
            size (int | tuple[int, int] | list[int], optional): Size of
            dropped out units. If int, all dropped out units have the same
            size. If list, a dropped out unit has size sampled from this list
            randomly. If 2-tuple, a dropped out unit has size sampled from
            this interval randomly. Note that dropped out units could overlap
            which results in larger units effectively, though the probability
            is low if `p` is small. Defaults to 1.
            fill (str | float, optional): How a dropped out value is filled.
            If "ffill", fill with the last previous value that is not dropped.
            If "bfill", fill with the first next value that is not dropped. If
            "mean", fill with the mean value of this channel in this series.
            If float, fill with this value. Defaults to "ffill".
            per_channel (bool, optional): Whether to sample dropout units
            independently for each channel in a time series or to use the same
            dropout units for all channels in a time series. Defaults to False.
        '''
        self.p = p
        self.fill = fill
        self.per_channel = per_channel

        if isinstance(size, int):
            self.size = [size]
        elif isinstance(size, tuple):
            self.size = list(range(size[0], size[1]))
        elif isinstance(size, list):
            self.size = size

    def __call__(self, x: np.ndarray) -> np.ndarray:
        '''`__call__` method.

        Args:
            x (numpy.ndarray): Input sequence to process.

        Returns:
            numpy.ndarray: Processed output sequence.
        '''
        L, C = x.shape

        if isinstance(self.p, (float, int)):
            p = np.ones(C if self.per_channel else 1) * self.p
        elif isinstance(self.p, tuple):
            p = np.random.uniform(p[0], p[1], C if self.per_channel else 1)
        elif isinstance(self.p, list):
            p = np.random.choice(self.p, C if self.per_channel else 1)

        x = x.swapaxes(0, 1)

        if isinstance(self.fill, str) and (self.fill == 'mean'):
            fill_value = x.mean(axis=0)

        for s in self.size:
            # sample dropout blocks
            if self.per_channel:
                drop = (
                    np.random.uniform(size=(C, L - s))
                    <= p.reshape(-1, 1) / len(self.size) / s
                )
            else:
                drop = (
                    np.random.uniform(size=(L - s))
                    <= p.reshape(-1, 1) / len(self.size) / s
                )
                drop = np.repeat(drop, C, axis=0)

            ind = np.argwhere(drop)  # position of dropout blocks

            if ind.size > 0:
                if isinstance(self.fill, str) and (self.fill == "ffill"):
                    i = np.repeat(ind[:, 0], s)
                    j0 = np.repeat(ind[:, 1], s)
                    j1 = j0 + np.tile(np.arange(1, s + 1), len(ind))
                    x[i, j1] = x[i, j0]
                elif isinstance(self.fill, str) and (self.fill == "bfill"):
                    i = np.repeat(ind[:, 0], s)
                    j0 = np.repeat(ind[:, 1], s) + s
                    j1 = j0 - np.tile(np.arange(1, s + 1), len(ind))
                    x[i, j1] = x[i, j0]
                elif isinstance(self.fill, str) and (self.fill == "mean"):
                    i = np.repeat(ind[:, 0], s)
                    j = np.repeat(ind[:, 1], s) + np.tile(
                        np.arange(1, s + 1), len(ind)
                    )
                    x[i, j] = fill_value[i]
                elif isinstance(self.fill, (float, int)):
                    i = np.repeat(ind[:, 0], s)
                    j = np.repeat(ind[:, 1], s) + np.tile(
                        np.arange(1, s + 1), len(ind)
                    )
                    x[i, j] = self.fill

        x = x.reshape(C, L).T

        return x


class TimeWarp:
    '''Random time warping. The augmenter random changed the speed of
    timeline. The time warping is controlled by the number of speed changes
    and the maximal ratio of max/min speed.'''

    def __init__(
        self,
        n_speed_change: int = 3,
        max_speed_ratio: float | tuple[float, float] | list[float] = 3.0,
    ) -> None:
        '''Random time warping. The augmenter random changed the speed of
        timeline. The time warping is controlled by the number of speed
        changes and the maximal ratio of max/min speed.

        Args:
            n_speed_change (int, optional): The number of speed changes in
            each series. Defaults to 3.
            max_speed_ratio (float | tuple[float, float] | list[float],
            optional): The maximal ratio of max/min speed in the warpped time
            line. The time line of a series is more likely to be significantly
            warpeped if this value is greater. If float, all series are
            warpped with the same ratio. If list, each series is warpped with
            a ratio that is randomly sampled from the list. If 2-tuple, each
            series is warpped with a ratio that is randomly sampled from the
            interval. Defaults to 3.0.
        '''
        self.n_speed_change = n_speed_change
        self.max_speed_ratio = max_speed_ratio

    def __call__(self, x: np.ndarray) -> np.ndarray:
        '''`__call__` method.

        Args:
            x (numpy.ndarray): Input sequence to process.

        Returns:
            numpy.ndarray: Processed output sequence.
        '''
        L, _ = x.shape

        idx = np.arange(L)
        anchors = np.arange(
            0,
            1 + 1 / (self.n_speed_change + 1) / 2,
            1 / (self.n_speed_change + 1),
        ) * (L - 1)

        if isinstance(self.max_speed_ratio, (float, int)):
            max_speed_ratio = np.array(self.max_speed_ratio)
        elif isinstance(self.max_speed_ratio, tuple):
            max_speed_ratio = np.random.uniform(
                low=self.max_speed_ratio[0], high=self.max_speed_ratio[1]
            )
        elif isinstance(self.max_speed_ratio, list):
            max_speed_ratio = np.random.choice(self.max_speed_ratio)

        anchor_values = np.random.uniform(
            low=0.0, high=1.0, size=self.n_speed_change + 1
        )
        anchor_values = anchor_values - (
            anchor_values.max() - max_speed_ratio * anchor_values.min()
        ) / (1 - max_speed_ratio)
        anchor_values = anchor_values.cumsum() / anchor_values.sum() * (L - 1)
        anchor_values = np.insert(anchor_values, 0, 0)

        warp = PchipInterpolator(x=anchors, y=anchor_values, axis=0)(idx)

        x = interp1d(
            idx, x, axis=0, fill_value='extrapolate', assume_sorted=True
        )(warp)

        return x
