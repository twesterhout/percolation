// Copyright (c) 2019, Tom Westerhout
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "particle.hpp"

#include "geometric_cluster.hpp"
#include <boost/container/static_vector.hpp>
#include <stack>

TCM_NAMESPACE_BEGIN

/// If root, destroys the geometric cluster and does nothing otherwise.
constexpr auto particle_base_t::destroy() noexcept -> void
{
    if (is_root()) { _cluster.~unique_ptr(); }
}

inline auto find_root(particle_base_t& particle) -> particle_base_t&
{
    TCM_ASSERT(particle.is_child(), "site is empty");
    using pointer_type = gsl::not_null<particle_base_t*>;
    using vector_type  = boost::container::static_vector<pointer_type, 16 - 1>;

    std::stack<pointer_type, vector_type> path;
    // Moving up the tree.
    auto p = pointer_type{std::addressof(particle)};
    while (!p->is_root()) {
        path.push(p);
        p = pointer_type{std::addressof(p->parent())};
    }
    auto& root = *p;

    // Path compression
    while (!path.empty()) {
        path.top()->parent(root);
        path.pop();
    }
    return root;
}

TCM_NAMESPACE_END
