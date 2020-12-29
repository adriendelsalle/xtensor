/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xadapt.hpp"

namespace xt
{

#define EXPECT_LAYOUT(EXPRESSION, LAYOUT)                         \
  EXPECT_TRUE((decltype(EXPRESSION)::static_layout == LAYOUT)) 


    TEST(xarray, asarray)
    {
        xt::xtensor<double, 1> arr {1., 2., 3.2};
        auto res1 = xt::asarray(arr);

        EXPECT_EQ(res1(2), 3.2);

        auto res2a = xt::asarray<double>(arr);
        EXPECT_EQ(res2a(2), 3.2);

        auto res2b = xt::asarray<double, layout_type::row_major>(arr);
        EXPECT_EQ(res2b(2), 3.2);

        auto res2c = xt::asarray<double, layout_type::row_major>(xt::xtensor<double, 1>({1., 2., 3.2}));
        EXPECT_EQ(res2c(2), 3.2);

        auto res3a = xt::asarray<int>(arr);
        EXPECT_EQ(res3a(2), 3);

        auto res3b = xt::asarray<int>(xt::xtensor<double, 1>({1., 2., 3.2}));
        EXPECT_LAYOUT(res3b, layout_type::row_major);
        EXPECT_EQ(res3b(2), 3);

        auto res4a = xt::asarray<int, layout_type::column_major>(xt::xtensor<double, 1>({1., 2., 3.2}));
        EXPECT_LAYOUT(res4a, layout_type::column_major);
        EXPECT_EQ(res4a(2), 3);

        auto res4b = xt::asarray<int, layout_type::column_major>(xt::xtensor<double, 2>({{1., 2., 3.2},
                                                                                         {4., 5., 6.}}));
        EXPECT_LAYOUT(res4b, layout_type::column_major);
        EXPECT_EQ(res4b(0, 2), 3);
    }

}
