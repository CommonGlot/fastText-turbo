/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "densematrix.h"

#include <random>
#include <stdexcept>
#include <thread>
#include <utility>
#include "utils.h"
#include "vector.h"

#if defined(__AVX512F__) || defined(__AVX__) || defined(__SSE__)
#include <immintrin.h>
#endif

namespace fasttext {

DenseMatrix::DenseMatrix() : DenseMatrix(0, 0) {}

DenseMatrix::DenseMatrix(int64_t m, int64_t n) : Matrix(m, n), data_(m * n) {}

DenseMatrix::DenseMatrix(DenseMatrix&& other) noexcept
    : Matrix(other.m_, other.n_), data_(std::move(other.data_)) {}

DenseMatrix::DenseMatrix(int64_t m, int64_t n, real* dataPtr)
    : Matrix(m, n), data_(dataPtr, dataPtr + (m * n)) {}

void DenseMatrix::zero() {
  std::fill(data_.begin(), data_.end(), 0.0);
}

void DenseMatrix::uniformThread(real a, int block, int32_t seed) {
  std::minstd_rand rng(block + seed);
  std::uniform_real_distribution<> uniform(-a, a);
  int64_t blockSize = (m_ * n_) / 10;
  for (int64_t i = blockSize * block;
       i < (m_ * n_) && i < blockSize * (block + 1);
       i++) {
    data_[i] = uniform(rng);
  }
}

void DenseMatrix::uniform(real a, unsigned int thread, int32_t seed) {
  if (thread > 1) {
    std::vector<std::thread> threads;
    for (int i = 0; i < thread; i++) {
      threads.push_back(std::thread([=]() { uniformThread(a, i, seed); }));
    }
    for (int32_t i = 0; i < threads.size(); i++) {
      threads[i].join();
    }
  } else {
    // webassembly can't instantiate `std::thread`
    uniformThread(a, 0, seed);
  }
}

void DenseMatrix::multiplyRow(const Vector& nums, int64_t ib, int64_t ie) {
  if (ie == -1) {
    ie = m_;
  }
  assert(ie <= nums.size());
  for (auto i = ib; i < ie; i++) {
    real n = nums[i - ib];
    if (n != 0) {
      for (auto j = 0; j < n_; j++) {
        at(i, j) *= n;
      }
    }
  }
}

void DenseMatrix::divideRow(const Vector& denoms, int64_t ib, int64_t ie) {
  if (ie == -1) {
    ie = m_;
  }
  assert(ie <= denoms.size());
  for (auto i = ib; i < ie; i++) {
    real n = denoms[i - ib];
    if (n != 0) {
      for (auto j = 0; j < n_; j++) {
        at(i, j) /= n;
      }
    }
  }
}

real DenseMatrix::l2NormRow(int64_t i) const {
  auto norm = 0.0;
  for (auto j = 0; j < n_; j++) {
    norm += at(i, j) * at(i, j);
  }
  if (std::isnan(norm)) {
    throw EncounteredNaNError();
  }
  return std::sqrt(norm);
}

void DenseMatrix::l2NormRow(Vector& norms) const {
  assert(norms.size() == m_);
  for (auto i = 0; i < m_; i++) {
    norms[i] = l2NormRow(i);
  }
}

real DenseMatrix::dotRow(const Vector& vec, int64_t i) const {
  assert(i >= 0);
  assert(i < m_);
  assert(vec.size() == n_);
  real d = 0.0;
  for (int64_t j = 0; j < n_; j++) {
    d += at(i, j) * vec[j];
  }
  if (std::isnan(d)) {
    throw EncounteredNaNError();
  }
  return d;
}

void DenseMatrix::multiplyVector(Vector& out, const Vector& vec) const {
#if defined(__clang__)
  // dotRow's sum is not fused into FMA instructions; do the same here so
  // the results stay identical (clang would otherwise fuse this loop).
#pragma clang fp contract(off)
#endif
  assert(out.size() == m_);
  assert(vec.size() == n_);
  // Eight rows per pass: each element of vec is loaded once for eight rows and
  // the eight sums are independent, so they overlap in the pipeline. Every
  // row is summed in the same order as dotRow, so the results are identical.
  constexpr int64_t kRows = 8;
  const real* v = vec.data();
  real* o = out.data();
  int64_t i = 0;
  for (; i + kRows <= m_; i += kRows) {
    const real* row = data_.data() + i * n_;
    real d[kRows] = {};
    for (int64_t j = 0; j < n_; j++) {
      const real x = v[j];
      for (int64_t r = 0; r < kRows; r++) {
        d[r] += row[r * n_ + j] * x;
      }
    }
    for (int64_t r = 0; r < kRows; r++) {
      o[i + r] = d[r];
    }
  }
  for (; i < m_; i++) {
    const real* row = data_.data() + i * n_;
    real d = 0.0;
    for (int64_t j = 0; j < n_; j++) {
      d += row[j] * v[j];
    }
    o[i] = d;
  }
  for (int64_t k = 0; k < m_; k++) {
    if (std::isnan(o[k])) {
      throw EncounteredNaNError();
    }
  }
}

void DenseMatrix::multiplyVectorFast(Vector& out, const Vector& vec) const {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  // x86: every dot product uses eight partial sums, which the compiler turns
  // into SIMD (SSE/AVX) operations, for eight rows per pass. The sums are
  // added in a different order than in dotRow, so scores differ from it by
  // float rounding only. (On ARM the exact multiplyVector is as fast.)
  assert(out.size() == m_);
  assert(vec.size() == n_);
  constexpr int64_t kRows = 8;
  constexpr int64_t kLanes = 8;
  const real* v = vec.data();
  real* o = out.data();
  const int64_t nv = n_ - n_ % kLanes;
  int64_t i = 0;
  for (; i + kRows <= m_; i += kRows) {
    const real* row = data_.data() + i * n_;
    real partial[kRows][kLanes] = {};
    for (int64_t j = 0; j < nv; j += kLanes) {
      for (int64_t r = 0; r < kRows; r++) {
        for (int64_t l = 0; l < kLanes; l++) {
          partial[r][l] += row[r * n_ + j + l] * v[j + l];
        }
      }
    }
    for (int64_t r = 0; r < kRows; r++) {
      real d = 0.0;
      for (int64_t l = 0; l < kLanes; l++) {
        d += partial[r][l];
      }
      for (int64_t j = nv; j < n_; j++) {
        d += row[r * n_ + j] * v[j];
      }
      o[i + r] = d;
    }
  }
  for (; i < m_; i++) {
    const real* row = data_.data() + i * n_;
    real partial[kLanes] = {};
    int64_t j = 0;
    for (; j < nv; j += kLanes) {
      for (int64_t l = 0; l < kLanes; l++) {
        partial[l] += row[j + l] * v[j + l];
      }
    }
    real d = 0.0;
    for (int64_t l = 0; l < kLanes; l++) {
      d += partial[l];
    }
    for (; j < n_; j++) {
      d += row[j] * v[j];
    }
    o[i] = d;
  }
  for (int64_t k = 0; k < m_; k++) {
    if (std::isnan(o[k])) {
      throw EncounteredNaNError();
    }
  }
#else
  multiplyVector(out, vec);
#endif
}

void DenseMatrix::addVectorToRow(const Vector& vec, int64_t i, real a) {
  assert(i >= 0);
  assert(i < m_);
  assert(vec.size() == n_);
  for (int64_t j = 0; j < n_; j++) {
    data_[i * n_ + j] += a * vec[j];
  }
}

void DenseMatrix::addRowToVector(Vector& x, int32_t i) const {
  assert(i >= 0);
  assert(i < this->size(0));
  assert(x.size() == this->size(1));
  for (int64_t j = 0; j < n_; j++) {
    x[j] += at(i, j);
  }
}

void DenseMatrix::addRowToVector(Vector& x, int32_t i, real a) const {
  assert(i >= 0);
  assert(i < this->size(0));
  assert(x.size() == this->size(1));
  for (int64_t j = 0; j < n_; j++) {
    x[j] += a * at(i, j);
  }
}

/* Abstract over AVX512F, AVX, and SSE intrinsics, using the one available on this machine. */
#if defined(__AVX512F__)
using Register = __m512;
inline Register Add(Register first, Register second) { return _mm512_add_ps(first, second); }
inline Register Set1(float to) { return _mm512_set1_ps(to); }
inline Register Multiply(Register first, Register second) { return _mm512_mul_ps(first, second); }
#elif defined(__AVX__)
using Register = __m256;
inline Register Add(Register first, Register second) { return _mm256_add_ps(first, second); }
inline Register Set1(float to) { return _mm256_set1_ps(to); }
inline Register Multiply(Register first, Register second) { return _mm256_mul_ps(first, second); }
#elif defined(__SSE__)
using Register = __m128;
inline Register Add(Register first, Register second) { return _mm_add_ps(first, second); }
inline Register Set1(float to) { return _mm_set1_ps(to); }
inline Register Multiply(Register first, Register second) { return _mm_mul_ps(first, second); }
#endif

/* Faster routine for averaging rows of a matrix on x86.
 * The idea here is to keep the accumulators in registers if possible. */
#if defined(__AVX512F__) || defined(__AVX__) || defined(__SSE__)
template <unsigned Cols> void averageRowsFast(Vector& x, const std::vector<int32_t>& rows, const DenseMatrix &matrix) {
  // Columns must be a multiple of how many floats fit in a register.
  static_assert(Cols % (sizeof(Register) / 4) == 0);
  constexpr unsigned RegisterCount = Cols / (sizeof(Register) / 4);
  // These should be aligned by aligned.h
  assert(reinterpret_cast<uintptr_t>(x.data()) % sizeof(Register) == 0);
  assert(reinterpret_cast<uintptr_t>(matrix.data()) % sizeof(Register) == 0);

  // Guard against empty list of rows with default NaN behavior.
  if (rows.empty()) {
    x.zero();
    x.mul(1.0 / rows.size());
    return;
  }

  // Copy the first row to accumulation registers.
  Register accum[RegisterCount];
  auto row = rows.cbegin();
  const Register *base = reinterpret_cast<const Register*>(matrix.data() + matrix.cols() * *row);
  for (unsigned i = 0; i < RegisterCount; ++i) {
    accum[i] = base[i];
  }
  // Add the rows after the first.
  for (++row; row != rows.cend(); ++row) {
    base = reinterpret_cast<const Register*>(matrix.data() + matrix.cols() * *row);
    for (unsigned i = 0; i < RegisterCount; ++i) {
      accum[i] = Add(accum[i], base[i]);
    }
  }
  // Multiply by (1.0 / rows.size()) and write to x.
  Register mul = Set1(1.0 / rows.size());
  for (unsigned i = 0; i < RegisterCount; ++i) {
    reinterpret_cast<Register*>(x.data())[i] = Multiply(accum[i], mul);
  }
}
#endif

#if !(defined(__AVX512F__) || defined(__AVX__) || defined(__SSE__))
/* Portable version of averageRowsFast: a fixed-size local accumulator that
 * the compiler can keep in vector registers (e.g. NEON on ARM). */
template <unsigned Cols> void averageRowsFast(Vector& x, const std::vector<int32_t>& rows, const DenseMatrix &matrix) {
  if (rows.empty()) {
    x.zero();
    x.mul(1.0 / rows.size());
    return;
  }
  real accum[Cols];
  const real* base = matrix.data() + matrix.cols() * rows[0];
  for (unsigned i = 0; i < Cols; ++i) {
    accum[i] = base[i];
  }
  for (size_t r = 1; r < rows.size(); ++r) {
    base = matrix.data() + matrix.cols() * rows[r];
    for (unsigned i = 0; i < Cols; ++i) {
      accum[i] += base[i];
    }
  }
  const real mul = 1.0 / rows.size();
  for (unsigned i = 0; i < Cols; ++i) {
    x[i] = accum[i] * mul;
  }
}
#endif

void DenseMatrix::averageRowsToVector(Vector& x, const std::vector<int32_t>& rows) const {
  switch (cols()) {
    case 512:
      // Maximum number that can fit all in registers on AVX512F.
      averageRowsFast<512>(x, rows, *this);
      return;
    case 256:
      averageRowsFast<256>(x, rows, *this);
      return;
    case 64:
      averageRowsFast<64>(x, rows, *this);
      return;
    case 32:
      averageRowsFast<32>(x, rows, *this);
      return;
    case 128:
      averageRowsFast<128>(x, rows, *this);
      return;
    case 16:
      averageRowsFast<16>(x, rows, *this);
      return;
  }
  x.zero();
  for (auto it = rows.cbegin(); it != rows.cend(); ++it) {
    addRowToVector(x, *it);
  }
  x.mul(1.0 / rows.size());
}

void DenseMatrix::save(std::ostream& out) const {
  out.write((char*)&m_, sizeof(int64_t));
  out.write((char*)&n_, sizeof(int64_t));
  out.write((char*)data_.data(), m_ * n_ * sizeof(real));
}

void DenseMatrix::load(std::istream& in) {
  in.read((char*)&m_, sizeof(int64_t));
  in.read((char*)&n_, sizeof(int64_t));
  data_ = intgemm::AlignedVector<real>(m_ * n_);
  in.read((char*)data_.data(), m_ * n_ * sizeof(real));
}

void DenseMatrix::dump(std::ostream& out) const {
  out << m_ << " " << n_ << std::endl;
  for (int64_t i = 0; i < m_; i++) {
    for (int64_t j = 0; j < n_; j++) {
      if (j > 0) {
        out << " ";
      }
      out << at(i, j);
    }
    out << std::endl;
  }
};

} // namespace fasttext
