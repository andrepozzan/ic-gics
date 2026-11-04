from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import logging
import os
from pathlib import PurePosixPath
import re
import struct
import time
from typing import Any
from urllib.parse import urljoin
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np


# RFWebLab operating limits and communication defaults.
SAMPLE_RATE = 200e6
MAX_SAMPLES = 1_000_000
MIN_SAMPLES = 1_000
MAX_PAPR_DB = 20.0
MAX_PEAK_DBM = -8.0
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_RESULT_BYTES = 32 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 30.0
POLL_INTERVAL_SECONDS = 10.0
MAX_POLL_ATTEMPTS = 100
UPLOAD_URL = "https://dpdcompetition.com/rfweblab/matlab/upload.php"
FILES_URL = "https://dpdcompetition.com/rfweblab/matlab/files/"
LOGGER = logging.getLogger("cadence_pa.rfweblab")


@dataclass(frozen=True)
class RFWebLabMeasurement:
    output: np.ndarray
    rms_output_dbm: float
    dc_current: float
    dc_voltage: float


def _environment_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _environment_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _power_dbm(signal: np.ndarray) -> float:
    return float(10.0 * np.log10(np.sum(np.abs(signal) ** 2) / 50.0 /
                                 signal.size) + 30.0)


def validate_input(signal: np.ndarray, rms_input_dbm: float) -> tuple[np.ndarray, float]:
    """Validate and normalize data using the limits enforced by RFWebLab."""
    values = np.asarray(signal, dtype=np.complex128).reshape(-1)
    if values.size < MIN_SAMPLES:
        raise ValueError(f"RFWebLab requires at least {MIN_SAMPLES} samples")
    if values.size > MAX_SAMPLES:
        values = values[:MAX_SAMPLES]
    if not np.all(np.isfinite(values)):
        raise ValueError("RFWebLab input contains NaN or Inf")
    if not np.isfinite(rms_input_dbm):
        raise ValueError("RFWebLab RMS input power must be finite")

    peak = max(float(np.max(np.abs(values.real))),
               float(np.max(np.abs(values.imag))))
    if peak:
        values = values / peak
    papr_db = 20.0 * np.log10(np.max(np.abs(values)) * np.sqrt(values.size) /
                               np.linalg.norm(values))
    if papr_db > MAX_PAPR_DB:
        raise ValueError(f"RFWebLab PAPR is {papr_db:.2f} dB; maximum is "
                         f"{MAX_PAPR_DB:.2f} dB")
    if rms_input_dbm > MAX_PEAK_DBM - papr_db * 0.77:
        raise ValueError("RFWebLab RMS input power exceeds the PAPR limit")
    if rms_input_dbm + papr_db > MAX_PEAK_DBM:
        raise ValueError("RFWebLab peak input power exceeds -8 dBm")
    return values, papr_db


def _serialize_matrix(values: Any) -> bytes:
    if isinstance(values, str):
        raw_values = np.frombuffer(values.encode("utf-8"), dtype="S1")
        return bytes([3, 2]) + struct.pack("<II", 1, raw_values.size) + raw_values.tobytes()
    array = np.asarray(values)
    if array.dtype.kind == "U":
        return _serialize_matrix(str(array.reshape(-1)[0]))
    class_code = {np.dtype("float64"): 0, np.dtype("float32"): 1,
                  np.dtype("int8"): 4, np.dtype("uint8"): 5,
                  np.dtype("int16"): 6, np.dtype("uint16"): 7,
                  np.dtype("int32"): 8, np.dtype("uint32"): 9,
                  np.dtype("int64"): 10, np.dtype("uint64"): 11}.get(
                      array.dtype, 0)
    raw = np.asarray(array, dtype=array.dtype.newbyteorder("<")).reshape(-1).tobytes()
    dimensions = array.shape or (1, 1)
    return bytes([class_code, len(dimensions)]) + b"".join(
        struct.pack("<I", dimension) for dimension in dimensions) + raw


def _serialize(value: Any) -> bytes:
    if isinstance(value, dict):
        fields = list(value)
        result = bytearray([255, len(fields)])
        for field in fields:
            encoded = field.encode("ascii")
            result.extend(struct.pack("<I", len(encoded)))
            result.extend(encoded)
        dimensions = (1, 1)
        result.extend([len(dimensions)])
        result.extend(b"".join(struct.pack("<I", n) for n in dimensions))
        for field in fields:
            result.extend(_serialize(value[field]))
        return bytes(result)
    return _serialize_matrix(value)


def _matlab_data_hash(serialized: bytes) -> bytes:
    # dataHash(uint8(serialized), 'MD5') hashes class, dimensions and bytes.
    prefix = b"uint8" + struct.pack("<QQQ", 2, len(serialized), 1)
    return hashlib.md5(prefix + serialized).digest()


def _deserialize(data: bytes, offset: int = 0) -> tuple[Any, int]:
    code = data[offset]
    offset += 1
    if code == 255:
        field_count = data[offset]
        offset += 1
        fields = []
        for _ in range(field_count):
            size = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            fields.append(data[offset:offset + size].decode("ascii"))
            offset += size
        dimensions, offset = _read_dimensions(data, offset)
        result = {}
        for field in fields:
            result[field], offset = _deserialize(data, offset)
        return result, offset
    if code == 254:
        dimensions, offset = _read_dimensions(data, offset)
        values = []
        for _ in range(int(np.prod(dimensions))):
            value, offset = _deserialize(data, offset)
            values.append(value)
        return values, offset
    dimensions, offset = _read_dimensions(data, offset)
    dtype = {0: "<f8", 1: "<f4", 2: "?", 4: "i1", 5: "u1",
             6: "<i2", 7: "<u2", 8: "<i4", 9: "<u4", 10: "<i8",
             11: "<u8"}.get(code)
    if code == 3:
        size = int(np.prod(dimensions))
        return data[offset:offset + size].decode("latin1"), offset + size
    if dtype is None:
        raise ValueError(f"Unknown RFWebLab serialized type {code}")
    size = int(np.prod(dimensions)) * np.dtype(dtype).itemsize
    values = np.frombuffer(data, dtype=dtype, count=int(np.prod(dimensions)),
                           offset=offset).copy()
    if code in (0, 1, 4, 5, 6, 7, 8, 9, 10, 11):
        values = values.astype(float)
    return values.reshape(dimensions), offset + size


def _read_dimensions(data: bytes, offset: int) -> tuple[tuple[int, ...], int]:
    count = data[offset]
    offset += 1
    dimensions = tuple(struct.unpack_from("<I", data, offset + 4 * index)[0]
                       for index in range(count))
    return dimensions, offset + 4 * count


class RFWebLabClient:
    def __init__(self, upload_url: str = UPLOAD_URL, files_url: str = FILES_URL,
                 timeout: float = REQUEST_TIMEOUT_SECONDS,
                 poll_interval: float = POLL_INTERVAL_SECONDS,
                 max_polls: int = MAX_POLL_ATTEMPTS):
        self.upload_url = upload_url
        self.files_url = files_url
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.max_polls = max_polls

    @classmethod
    def from_environment(cls) -> "RFWebLabClient":
        return cls(
            upload_url=os.getenv("RFWEBLAB_UPLOAD_URL", UPLOAD_URL),
            files_url=os.getenv("RFWEBLAB_FILES_URL", FILES_URL),
            timeout=_environment_float(
                "RFWEBLAB_TIMEOUT_SECONDS", REQUEST_TIMEOUT_SECONDS),
            poll_interval=_environment_float(
                "RFWEBLAB_POLL_INTERVAL_SECONDS", POLL_INTERVAL_SECONDS),
            max_polls=_environment_int(
                "RFWEBLAB_MAX_POLL_ATTEMPTS", MAX_POLL_ATTEMPTS),
        )

    def measure(self, signal: np.ndarray, rms_input_dbm: float) -> RFWebLabMeasurement:
        values, papr_db = validate_input(signal, rms_input_dbm)
        LOGGER.info(
            "RFWebLab validation passed: samples=%d, rms_input=%.3f dBm, "
            "papr=%.3f dB, upload_url=%s",
            values.size, rms_input_dbm, papr_db, self.upload_url,
        )
        payload = _serialize({
            "Client_version": np.asarray(1.2, dtype=np.float32),
            "OpMode": "A5Z6UNud",
            "PowerLevel": np.asarray(rms_input_dbm),
            "SA_Samples": np.asarray(values.size, dtype=np.float64),
            "Re_data": values.real.reshape(-1, 1),
            "Im_data": values.imag.reshape(-1, 1),
        })
        request_payload = _matlab_data_hash(payload) + payload
        LOGGER.info(
            "RFWebLab request prepared: payload_bytes=%d, iq_bytes=%d, "
            "timeout=%.1fs",
            len(request_payload), values.nbytes * 2, self.timeout,
        )
        response = self._upload(request_payload)
        filename = self._parse_filename(response)
        LOGGER.info("RFWebLab upload accepted: output_file=%s", filename)
        output = self._poll(filename)
        LOGGER.info("RFWebLab response downloaded: bytes=%d", len(output))
        parsed, _ = _deserialize(output)
        try:
            status = int(np.asarray(parsed["status"]).reshape(-1)[0])
            error_code = int(np.asarray(parsed["error_hanlde"]).reshape(-1)[0])
            LOGGER.info(
                "RFWebLab measurement response: status=%d, error_code=%d",
                status, error_code,
            )
            if status <= 0:
                raise RuntimeError(
                    f"RFWebLab rejected the measurement (error {error_code})")
            result = RFWebLabMeasurement(
                output=np.asarray(parsed["b3_re"]).reshape(-1) +
                1j * np.asarray(parsed["b3_im"]).reshape(-1),
                rms_output_dbm=float(np.asarray(parsed["RMS_out"]).reshape(-1)[0]),
                dc_current=float(np.asarray(parsed["DCmeas"]["Id"]).reshape(-1)[0]),
                dc_voltage=float(np.asarray(parsed["DCmeas"]["Vd"]).reshape(-1)[0]),
            )
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ValueError("RFWebLab response is missing expected fields") from error
        if result.output.size != values.size:
            raise ValueError("RFWebLab returned an unexpected sample count")
        return result

    def _upload(self, payload: bytes) -> str:
        boundary = b"rfweblab-python-boundary"
        body = (b"--" + boundary + b"\r\n"
                b'Content-Disposition: form-data; name="myFile"; filename="input.dat"\r\n'
                b"Content-Type: application/octet-stream\r\n\r\n" + payload +
                b"\r\n--" + boundary + b"--\r\n")
        request = Request(self.upload_url, data=body, method="POST",
                          headers={"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"})
        started = time.monotonic()
        LOGGER.info("RFWebLab HTTP POST started: url=%s, body_bytes=%d",
                    self.upload_url, len(body))
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read()
                LOGGER.info(
                    "RFWebLab HTTP POST completed: status=%d, response_bytes=%d, "
                    "elapsed=%.3fs",
                    response.status, len(body), time.monotonic() - started,
                )
                return body.decode("utf-8", errors="strict")
        except (HTTPError, URLError, TimeoutError, OSError):
            LOGGER.exception(
                "RFWebLab HTTP POST failed after %.3fs", time.monotonic() - started)
            raise

    @staticmethod
    def _parse_filename(response: str) -> str:
        match = re.search(r"(output_[A-Za-z0-9_.-]+\.dat)", response)
        if not match:
            raise ValueError("RFWebLab upload response did not contain an output file")
        return match.group(1)

    def _poll(self, filename: str) -> bytes:
        zip_url = urljoin(self.files_url, filename[:-3] + "zip")
        data_url = urljoin(self.files_url, filename)
        ready_url = urljoin(self.files_url, "ok_" + filename[7:])
        for _ in range(self.max_polls):
            attempt = _ + 1
            LOGGER.info(
                "RFWebLab polling attempt %d/%d: ready_url=%s",
                attempt, self.max_polls, ready_url,
            )
            try:
                with urlopen(ready_url, timeout=self.timeout) as ready_response:
                    LOGGER.info(
                        "RFWebLab ready marker found: status=%d, bytes=%s; "
                        "final_zip=%s; final_data=%s",
                        ready_response.status,
                        ready_response.headers.get("Content-Length", "?"),
                        zip_url,
                        data_url,
                    )
                with urlopen(zip_url, timeout=self.timeout) as response:
                    archive = response.read()
                    LOGGER.info("RFWebLab ZIP downloaded: status=%d, bytes=%d",
                                response.status, len(archive))
                if len(archive) > MAX_ARCHIVE_BYTES:
                    raise ValueError("RFWebLab archive exceeds the safe size limit")
                with ZipFile(io.BytesIO(archive)) as zipped:
                    members = [name for name in zipped.namelist()
                               if PurePosixPath(name).name == filename]
                    if len(members) != 1:
                        raise ValueError("RFWebLab archive has no unique output data file")
                    info = zipped.getinfo(members[0])
                    if info.file_size > MAX_RESULT_BYTES:
                        raise ValueError("RFWebLab result exceeds the safe size limit")
                    return zipped.read(members[0])
            except HTTPError as error:
                if error.code in (400, 404):
                    LOGGER.info(
                        "RFWebLab result is not ready yet: HTTP %d; "
                        "sleeping %.1fs",
                        error.code, self.poll_interval,
                    )
                else:
                    LOGGER.warning("RFWebLab polling HTTP error %d: %s",
                                   error.code, error.reason)
                time.sleep(self.poll_interval)
            except (URLError, TimeoutError, OSError, ValueError) as error:
                LOGGER.warning("RFWebLab polling failed temporarily: %s", error)
                time.sleep(self.poll_interval)
        raise TimeoutError("RFWebLab measurement did not finish before timeout")
