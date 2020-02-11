#ifndef M_M_HPP
#define M_M_HPP

class MM
{
    public:  
        const vector <vector <double> > mMultiply(const vector <vector <double> > &v1, const vector <vector <double> > &v2)
        {
            assert(v1.size() == v2.size() && v1[0].size() == v2[0].size() && "Vectors' dimension do not match!");
            vector <vector <double> > item;

            for (int r = 0; r < v1.size(); r ++)
            {
                vector <double> entry;

                for (int c = 0; c < v1[r].size(); c ++)
                {
                    entry.push_back(v1[r][c] * v2[r][c]);
                }
                item.push_back(entry);
            }

            return item;

        }

        const vector <vector <double> > mSubtract(const double &target, const vector <vector <double> > &v1)
        {
            assert(v1.size() > 0 && "Vector is Empty!");
            vector <vector <double> > item;

            for (auto &row : v1)
            {
                vector <double> entry;

                for (auto &col : row)
                {
                    entry.push_back(target - col);
                }

                item.push_back(entry);
            }
            return item;
        }

        const vector <vector <double> > transpose(const vector <vector <double> > &v1)
        {
            assert(v1.size() > 0 && "Vector is Empty!");
            vector <vector <double> > item;

            for (int c = 0; c < v1[0].size(); c++)
            {
                vector <double> entry;
                for (auto &row : v1)
                {
                    entry.push_back(row[c]);
                }
                item.push_back(entry);
            }

            return item;
        }

        void addItem(vector <vector <double> > &v1, const int &rowID, const double &target)
        {
            for (auto &col : v1[rowID])
            {
                col += target;
            }
        }

        const vector <double> ravel (const vector <vector <double> > &v1)
        {
            assert(v1.size() > 0 && "Vector is empty!");

            vector <double> item;
            for (auto &row: v1)
            {
                for (auto &col: row)
                {
                    item.push_back(col);
                }
            }
            return item;
        }

        void setItem(vector <vector <double> > &v1, const int &rowID, const double &target)
        {
            for (auto &col : v1[rowID])
            {
                col = target;
            }
        }
   
        const vector <vector <double> > mDivide(vector <vector <double> > &v1, const double &target)
        {
            /* Divide all elements with target */
            assert(v1.size() > 0 && "Vector is empty!");
            vector <vector <double> > item;
            for (auto &row : v1)
            {
                vector <double> elements;
                for (auto &col: row)
                {
                    elements.push_back(col / target);
                }
                item.push_back(elements);
            }
            return item;
        }

        const double mSum(const vector <vector <double> > &v1)
        {
            /* Find the sum of vector */
            assert(v1.size() > 0 && "Vector is empty!");
            double sum = 0.0;

            for (auto &row: v1)
            {
                for (auto &col: row)
                {
                    sum += col;
                }
            }

            return sum;
        }
        
        const vector < vector <double> > mExp(const vector <vector <double> > &v1)
        {
            /* Perform exp(x) on Matrix */
            assert(v1.size() > 0 && "Vector is empty!");
            vector <vector <double> > item;

            for (int r = 0; r < v1.size(); r ++)
            {
                vector <double> elements;
                for (int c = 0; c < v1[0].size(); c ++)
                {
                    elements.push_back(exp(v1[r][c]));
                }
                item.push_back(elements);
            }

            return item;
        }

        const vector < vector <double> > mTanh(const vector <vector <double> > &v1)
        {
            /* Perform Tan(x) on Matrix */
            assert(v1.size() > 0 && "Vector is empty!");
            vector <vector <double> > item;

            for (int r = 0; r < v1.size(); r ++)
            {
                vector <double> elements;
                for (int c = 0; c < v1[0].size(); c ++)
                {
                    elements.push_back(tan(v1[r][c]));
                }
                item.push_back(elements);
            }

            return item;

        }

};

#endif 