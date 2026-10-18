/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <istream>
#include <ostream>
#include <vector>

#include <assert.h>
#include "real.h"

namespace fasttext {

class Vector;

class Matrix {
 protected:
  int64_t m_;
  int64_t n_;

 public:
  Matrix();
  explicit Matrix(int64_t, int64_t);
  virtual ~Matrix() = default;

  int64_t size(int64_t dim) const;

  virtual real dotRow(const Vector&, int64_t) const = 0;
  // out[i] = dotRow(vec, i) for every row; subclasses may compute several
  // rows per pass.
  virtual void multiplyVector(Vector& out, const Vector& vec) const;
  // Like multiplyVector, but each dot product may be summed in another order
  // if that is faster on this platform (results within float rounding).
  virtual void multiplyVectorFast(Vector& out, const Vector& vec) const;
  virtual void addVectorToRow(const Vector&, int64_t, real) = 0;
  virtual void addRowToVector(Vector& x, int32_t i) const = 0;
  virtual void addRowToVector(Vector& x, int32_t i, real a) const = 0;
  virtual void averageRowsToVector(Vector& x, const std::vector<int32_t>& rows) const = 0;
  virtual void save(std::ostream&) const = 0;
  virtual void load(std::istream&) = 0;
  virtual void dump(std::ostream&) const = 0;
};

} // namespace fasttext
