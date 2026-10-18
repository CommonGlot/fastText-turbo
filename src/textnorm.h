/**
 * Copyright (c) 2026 CommonGlot contributors.
 * Licensed under the MIT license; see LICENSE.
 *
 * Lowercasing and the CommonLID-style input normalization used by the
 * -normProb training option.
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "norm_table.h"
#include "unicode_fold.h"

namespace fasttext {

namespace text {

inline bool isContinuation(uint8_t b) {
  return (b & 0xC0) == 0x80;
}

inline char asciiLower(char c) {
  return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + ('a' - 'A')) : c;
}

inline uint32_t lowercase(uint32_t cp) {
  int lo = 0, hi = unicode::kLowercaseSize - 1;
  while (lo <= hi) {
    const int mid = (lo + hi) / 2;
    const uint32_t from = unicode::kLowercase[mid].from;
    if (from == cp) {
      return unicode::kLowercase[mid].to;
    }
    if (from < cp) {
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return cp;
}

inline void appendUtf8(uint32_t cp, std::string& out) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

// Length of the well-formed UTF-8 sequence starting at s[i], or 0.
inline size_t utf8Length(std::string_view s, size_t i, uint32_t& cp) {
  const uint8_t b = static_cast<uint8_t>(s[i]);
  size_t len;
  if (b >= 0xC2 && b <= 0xDF) {
    len = 2;
    cp = b & 0x1F;
  } else if (b >= 0xE0 && b <= 0xEF) {
    len = 3;
    cp = b & 0x0F;
  } else if (b >= 0xF0 && b <= 0xF4) {
    len = 4;
    cp = b & 0x07;
  } else {
    return 0;
  }
  if (i + len > s.size()) {
    return 0;
  }
  for (size_t k = 1; k < len; k++) {
    const uint8_t c = static_cast<uint8_t>(s[i + k]);
    if (!isContinuation(c)) {
      return 0;
    }
    cp = (cp << 6) | (c & 0x3F);
  }
  return len;
}

// Lowercase s in place (for normalizeToken). Unicode lowercasing uses
// one-to-one mappings only, so the number of code points never changes;
// malformed bytes are kept as is.
inline void lowercaseUtf8(std::string& s) {
  bool ascii = true;
  for (char& c : s) {
    c = asciiLower(c);
    ascii = ascii && static_cast<uint8_t>(c) < 0x80;
  }
  if (ascii) {
    return;
  }
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size();) {
    if (static_cast<uint8_t>(s[i]) < 0x80) {
      out.push_back(s[i++]);
      continue;
    }
    uint32_t cp = 0;
    const size_t len = utf8Length(s, i, cp);
    if (len == 0) {
      out.push_back(s[i++]);
      continue;
    }
    const uint32_t lower = lowercase(cp);
    if (lower == cp) {
      out.append(s, i, len);
    } else {
      appendUtf8(lower, out);
    }
    i += len;
  }
  s.swap(out);
}

// Whether the OpenLID-v2 / CommonLID cleaner deletes code point cp
// (non-word characters and digits; see tools/gen_norm_table.py).
inline bool normRemoved(uint32_t cp) {
  int lo = 0, hi = unicode::kNormRemovedSize - 1;
  while (lo <= hi) {
    const int mid = (lo + hi) / 2;
    if (cp < unicode::kNormRemoved[mid].first) {
      hi = mid - 1;
    } else if (cp > unicode::kNormRemoved[mid].last) {
      lo = mid + 1;
    } else {
      return true;
    }
  }
  return false;
}

// Normalize one token like the CommonLID input cleaner: lowercase, then drop
// non-word characters and digits. Malformed UTF-8 bytes are dropped too.
inline void normalizeToken(std::string& s) {
  // Python lowercases U+0130 (capital I with dot) to "i" + U+0307; match it.
  for (size_t pos = s.find("\xC4\xB0"); pos != std::string::npos;
       pos = s.find("\xC4\xB0", pos + 3)) {
    s.replace(pos, 2, "i\xCC\x87");
  }
  lowercaseUtf8(s);
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size();) {
    const uint8_t b = static_cast<uint8_t>(s[i]);
    if (b < 0x80) {
      if (!normRemoved(b)) {
        out.push_back(s[i]);
      }
      i++;
      continue;
    }
    uint32_t cp = 0;
    const size_t len = utf8Length(s, i, cp);
    if (len == 0) {
      i++;
      continue;
    }
    if (!normRemoved(cp)) {
      out.append(s, i, len);
    }
    i += len;
  }
  s.swap(out);
}

} // namespace text
} // namespace fasttext
