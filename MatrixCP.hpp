#ifndef MCP_HPP
#define MCP_HPP
#include <vector>
using namespace std;
/* Matrices Cross Product */

class MatrixCP 
{
    private: 
        const vector < vector <double> > dekaCP (const vector <vector <double> > &v1, const vector <vector <double> > &v2)
        {
            /* Perform Matrices Multiplication; Dot Product */
            vector <vector <double> > item;
            
            //start with the row in v1
            for (auto &row : v1)
            {
                vector <double> entries;

                for (int v2_col = 0; v2_col < v2[0].size(); v2_col ++)
                {
                    double sum = 0.0;

                    for (int col = 0; col < row.size()-9; col += 10)
                    {
                        sum += row[col] * v2[col][v2_col] + row[col + 1] * v2[col + 1][v2_col]
                            + row[col + 2] * v2[col + 2][v2_col] + row[col + 3] * v2[col + 3][v2_col]
                            + row[col + 4] * v2[col + 4][v2_col] + row[col + 5] * v2[col + 5][v2_col]
                            + row[col + 6] * v2[col + 6][v2_col] + row[col + 7] * v2[col + 7][v2_col]
                            + row[col + 8] * v2[col + 8][v2_col] + + row[col + 9] * v2[col + 9][v2_col];
                    }
                    entries.push_back(sum);
                }
                item.push_back(entries);
            }
            return item;
        }

        const vector < vector <double> > etaCP (const vector <vector <double> > &v1, const vector <vector <double> > &v2)
        {
            /* Perform Matrices Multiplication; Dot Product */
            vector <vector <double> > item;
            
            //start with the row in v1
            for (auto &row : v1)
            {
                vector <double> entries;

                for (int v2_col = 0; v2_col < v2[0].size(); v2_col ++)
                {
                    double sum = 0.0;

                    for (int col = 0; col < row.size()-6; col += 7)
                    {
                        sum += row[col] * v2[col][v2_col] + row[col + 1] * v2[col + 1][v2_col]
                            + row[col + 2] * v2[col + 2][v2_col] + row[col + 3] * v2[col + 3][v2_col]
                            + row[col + 4] * v2[col + 4][v2_col] + row[col + 5] * v2[col + 5][v2_col]
                            + row[col + 6] * v2[col + 6][v2_col];
                    }
                    entries.push_back(sum);
                }
                item.push_back(entries);
            }
            return item;
        }

        const vector < vector <double> > pentaCP (const vector <vector <double> > &v1, const vector <vector <double> > &v2)
        {
            /* Perform Matrices Multiplication; Dot Product */
            vector <vector <double> > item;
            
            //start with the row in v1
            for (auto &row : v1)
            {
                vector <double> entries;

                for (int v2_col = 0; v2_col < v2[0].size(); v2_col ++)
                {
                    double sum = 0.0;

                    for (int col = 0; col < row.size()-4; col += 5)
                    {
                        sum += row[col] * v2[col][v2_col] + row[col + 1] * v2[col + 1][v2_col]
                            + row[col + 2] * v2[col + 2][v2_col] + row[col + 3] * v2[col + 3][v2_col]
                            + row[col + 4] * v2[col + 4][v2_col];
                    }
                    entries.push_back(sum);
                }
                item.push_back(entries);
            }
            return item;
        }

        const vector < vector <double> > quadraCP (const vector <vector <double> > &v1, const vector <vector <double> > &v2)
        {
            /* Perform Matrices Multiplication; Dot Product */
            vector <vector <double> > item;
            
            //start with the row in v1
            for (auto &row : v1)
            {
                vector <double> entries;

                for (int v2_col = 0; v2_col < v2[0].size(); v2_col ++)
                {
                    double sum = 0.0;

                    for (int col = 0; col < row.size()-3; col += 4)
                    {
                        sum += row[col] * v2[col][v2_col] + row[col + 1] * v2[col + 1][v2_col]
                            + row[col + 2] * v2[col + 2][v2_col] + row[col + 3] * v2[col + 3][v2_col];
                           
                    }
                    entries.push_back(sum);
                }
                item.push_back(entries);
            }
            return item;
        }

        const vector < vector <double> > doubleCP (const vector <vector <double> > &v1, const vector <vector <double> > &v2)
        {
            /* Perform Matrices Multiplication; Dot Product */
            vector <vector <double> > item;
            
            //start with the row in v1
            for (auto &row : v1)
            {
                vector <double> entries;

                for (int v2_col = 0; v2_col < v2[0].size(); v2_col ++)
                {
                    double sum = 0.0;

                    for (int col = 0; col < row.size()-1; col += 2)
                    {
                        sum += row[col] * v2[col][v2_col] + row[col + 1] * v2[col+1][v2_col];
                    }
                    entries.push_back(sum);
                }
                item.push_back(entries);
            }
            return item;
        }

        const vector <vector <double> > singleCP(const vector <vector <double> > &v1, const vector <vector <double> > &v2)
        {
            /* Matrix Multiplication single iterative */
            vector <vector <double> > item;

            for (auto &row : v1)
            {
                vector <double> entries;

                for (int v2_col = 0; v2_col < v2[0].size(); v2_col ++)
                {
                    double sum = 0.0;

                    for (int col = 0; col < row.size()-1; col ++)
                    {
                        sum += row[col] * v2[col][v2_col];
                    }
                    entries.push_back(sum);
                }
                item.push_back(entries);
            }
            return item;
        }

    public:     
        const vector <vector <double> > cross_product(const vector <vector <double> > &v1, const vector <vector <double> > &v2)
        {
            assert (v1[0].size() == v2.size() && "Vector 1's column size does not match Vector 2's row size!!!");
            
            int target_size = v1[0].size();

            if (target_size > 10)
            {
                if (target_size % 10 == 0)
                {
                    return dekaCP(v1, v2);
                }

                else if (target_size % 7 == 0)
                {
                    return etaCP(v1, v2);
                }
                else if (target_size % 5 == 0)
                {
                    return pentaCP(v1, v2);
                }
                else if (target_size % 4 == 0)
                {
                    return quadraCP(v1, v2);
                }
                else if (target_size % 2 == 0)
                {
                    return doubleCP(v1, v2);
                }
                else
                {
                    return singleCP(v1, v2);
                }
                
            }
            else 
            {
                return singleCP(v1, v2);
            }
        }
};
#endif