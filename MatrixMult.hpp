#ifndef M_MUL_HPP
#define M_MUL_HPP

class MatrixMult
{
    private: 
        const vector <vector <double> > dekaMult(const vector <vector <double> > &v1, const vector <vector <double> > &v2)
        {
            vector <vector <double> > item;

            for (int r = 0; r < v1.size(); r ++)
            {
                vector <double> elements;
                for (int c = 0; c < v1[0].size()-9; c += 10)
                {
                    elements.push_back(v1[r][c] * v2[r][c]);
                    elements.push_back(v1[r][c + 1] * v2[r][c + 1]);
                    elements.push_back(v1[r][c + 2] * v2[r][c + 2]);
                    elements.push_back(v1[r][c + 3] * v2[r][c + 3]);
                    elements.push_back(v1[r][c + 4] * v2[r][c + 4]);
                    elements.push_back(v1[r][c + 5] * v2[r][c + 5]);
                    elements.push_back(v1[r][c + 6] * v2[r][c + 6]);
                    elements.push_back(v1[r][c + 7] * v2[r][c + 7]);
                    elements.push_back(v1[r][c + 8] * v2[r][c + 8]);
                    elements.push_back(v1[r][c + 9] * v2[r][c + 9]);
                }
                item.push_back(elements);
            }
            return item;
        }

        const vector <vector <double> > etaMult(const vector <vector <double> > &v1, const vector <vector <double> > &v2)
        {
            vector <vector <double> > item;

            for (int r = 0; r < v1.size(); r ++)
            {
                vector <double> elements;
                for (int c = 0; c < v1[0].size()-6; c += 7)
                {
                    elements.push_back(v1[r][c] * v2[r][c]);
                    elements.push_back(v1[r][c + 1] * v2[r][c + 1]);
                    elements.push_back(v1[r][c + 2] * v2[r][c + 2]);
                    elements.push_back(v1[r][c + 3] * v2[r][c + 3]);
                    elements.push_back(v1[r][c + 4] * v2[r][c + 4]);
                    elements.push_back(v1[r][c + 5] * v2[r][c + 5]);
                    elements.push_back(v1[r][c + 6] * v2[r][c + 6]);
                }
                item.push_back(elements);
            }
            return item;
        }

        const vector <vector <double> > pentaMult(const vector <vector <double> > &v1, const vector <vector <double> > &v2)
        {
            vector <vector <double> > item;

            for (int r = 0; r < v1.size(); r ++)
            {
                vector <double> elements;
                for (int c = 0; c < v1[0].size()-4; c += 5)
                {
                    elements.push_back(v1[r][c] * v2[r][c]);
                    elements.push_back(v1[r][c + 1] * v2[r][c + 1]);
                    elements.push_back(v1[r][c + 2] * v2[r][c + 2]);
                    elements.push_back(v1[r][c + 3] * v2[r][c + 3]);
                    elements.push_back(v1[r][c + 4] * v2[r][c + 4]);
                }
                item.push_back(elements);
            }
            return item;
        }

        const vector <vector <double> > quadraMult(const vector <vector <double> > &v1, const vector <vector <double> > &v2)
        {
            vector <vector <double> > item;

            for (int r = 0; r < v1.size(); r ++)
            {
                vector <double> elements;
                for (int c = 0; c < v1[0].size()-3; c += 4)
                {
                    elements.push_back(v1[r][c] * v2[r][c]);
                    elements.push_back(v1[r][c + 1] * v2[r][c + 1]);
                    elements.push_back(v1[r][c + 2] * v2[r][c + 2]);
                    elements.push_back(v1[r][c + 3] * v2[r][c + 3]);
                }
                item.push_back(elements);
            }
            return item;
        }

        const vector <vector <double> > doubleMult(const vector <vector <double> > &v1, const vector <vector <double> > &v2)
        {
            vector <vector <double> > item;

            for (int r = 0; r < v1.size(); r ++)
            {
                vector <double> elements;
                for (int c = 0; c < v1[0].size()-1; c += 2)
                {
                    elements.push_back(v1[r][c] * v2[r][c]);
                    elements.push_back(v1[r][c + 1] * v2[r][c + 1]);
                }
                item.push_back(elements);
            }
            return item;
        }

        const vector <vector <double> > singleMult(const vector <vector <double> > &v1, const vector <vector <double> > &v2)
        {
            vector <vector <double> > item;

            for (int r = 0; r < v1.size(); r ++)
            {
                vector <double> elements;
                for (int c = 0; c < v1[0].size(); c ++)
                {
                    elements.push_back(v1[r][c] * v2[r][c]);
                }
                item.push_back(elements);
            }
            return item;
        }
    
    public:    
        const vector <vector <double> > matrix_mult(const vector <vector <double> > &v1, const vector <vector <double> > &v2)
        {
            assert(v1.size() == v2.size() && v1[0].size() == v2[0].size() && "The size of the vectors do not match!");
            
            int target_size = v1[0].size();

            if (target_size > 10)
            {
                if (target_size % 10 == 0)
                {
                    return dekaMult(v1, v2);
                }
                else if (target_size % 7 == 0)
                {
                    return etaMult(v1, v2);
                }
                else if (target_size % 5 == 0)
                {
                    return pentaMult(v1, v2);
                }
                else if (target_size % 4 == 0)
                {
                    return quadraMult(v1, v2);
                }
                else if (target_size % 2 == 0)
                {
                    return doubleMult(v1, v2);
                }
                else 
                {
                    return singleMult(v1, v2);
                }
            }
            else 
            {
                return singleMult(v1, v2);
            }
        }
};

#endif