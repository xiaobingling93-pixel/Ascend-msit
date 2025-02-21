# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from components.profile.msprof.ait_prof.api import get_csv_to_df, get_nlargest, get_top3_sum, get_nhead, \
    get_value_from_str, get_value_from_index, get_label_and_content, get_index_and_row, check_value, \
    get_bigger_value, get_smaller_value, get_item_value, set_item_value, get_col_value, \
    set_col_value, add_row, insert_row_1, drop_row_1, drop_row_2

class TestDataFrameUtils(unittest.TestCase):
    def test_get_nlargest(self):
        df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        result = get_nlargest(df, 3, 'A')
        self.assertTrue(result.equals(df.nlargest(3, 'A')))

    def test_get_top3_sum(self):
        df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        result = get_top3_sum(df, 'A')
        self.assertEqual(result, df['A'].nlargest(3).sum())

    def test_get_nhead(self):
        df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        result = get_nhead(df, 3)
        self.assertTrue(result.equals(df.head(3)))

    def test_get_value_from_str(self):
        # 创建DataFrame时，明确指定索引和列的标签
        df = pd.DataFrame({'A': [1, 2, 3]}, index=['row1', 'row2', 'row3'])
        
        # 由于现在行和列都有标签，可以直接使用.loc
        result = get_value_from_str(df, 'row1', 'A')
        self.assertEqual(result, df.loc['row1', 'A'])

    def test_get_value_from_index(self):
        df = pd.DataFrame({'A': [1, 2, 3]})
        result = get_value_from_index(df, 0, 0)
        self.assertEqual(result, df.iat[0, 0])

    def test_get_label_and_content(self):
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        label_list, _ = get_label_and_content(df)
        self.assertEqual(label_list, df.columns.tolist())

    def test_get_index_and_row(self):
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        index_list, _ = get_index_and_row(df)
        self.assertEqual(index_list, df.index.tolist())

    def test_check_value(self):
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        value = {'A': [1]}
        result = check_value(df, value)
        self.assertTrue(result.equals(df.isin(value)))

    def test_get_bigger_value(self):
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        result = get_bigger_value(df, 'A', 1)
        self.assertTrue(result.equals(df.where(df['A'] > 1)))

    def test_get_smaller_value(self):
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        result = get_smaller_value(df, 'A', 2)
        self.assertTrue(result.equals(df.where(df['A'] < 2)))

    def test_get_item_value(self):
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        result = get_item_value(df, 'A', 2, 'B')
        self.assertEqual(result, df.at[df.index[df['A'] == 2].values[-1], 'B'])

    def test_set_item_value(self):
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        set_item_value(df, 'A', 2, 'B', 5)
        self.assertEqual(df.at[df.index[df['A'] == 2].values[-1], 'B'], 5)

    def test_get_col_value(self):
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        result = get_col_value(df, 'A', 2, 'B')
        self.assertEqual(result.tolist(), df.loc[df['A'] == 2, 'B'].values.tolist())

    def test_set_col_value(self):
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        set_col_value(df, 'A', 2, 'B', 5)
        self.assertTrue(df.equals(pd.DataFrame({'A': [1, 2], 'B': [3, 5]})))

    def test_add_row(self):
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        rows = [{'A': 3, 'B': 5}]
        result = add_row(df, rows)
        self.assertTrue(result.equals(df.append(pd.DataFrame(rows), ignore_index=True)))

    def test_insert_row_1(self):
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        rows = [{'A': 3, 'B': 5}]
        result = insert_row_1(df, 'A', 2, rows)
        self.assertTrue(result.equals(pd.DataFrame({'A': [1, 2, 3], 'B': [3, 4, 5]})))

    def test_drop_row_1(self):
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        drop_row_1(df, 'A', 2)
        self.assertTrue(df.equals(pd.DataFrame({'A': [1], 'B': [3]})))

    def test_drop_row_2(self):
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        drop_row_2(df, 'A', 2, 'B', 4)
        self.assertTrue(df.equals(pd.DataFrame({'A': [1], 'B': [3]})))

if __name__ == '__main__':
    unittest.main()