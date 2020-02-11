#ifndef M_ADD_HPP
#define M_ADD_HPP

class MatrixAdd
{
    private: 
        const vector <vector <double> > dekaADD(const vector <vector <double> > &v1, const vector <vector <double> > &v2)
        {
            vector <vector <double> > item;

            for (int r = 0; r < v1.size(); r ++)
            {
                vector <double> elements;
                for (int c = 0; c < v1[0].size()-9; c += 10)
                {
                    elements.push_back(v1[r][c] + v2[r][c]);
                    elements.push_back(v1[r][c + 1] + v2[r][c + 1]);
                    elements.push_back(v1[r][c + 2] + v2[r][c + 2]);
                    elements.push_back(v1[r][c + 3] + v2[r][c + 3]);
                    elements.push_back(v1[r][c + 4] + v2[r][c + 4]);
                    elements.push_back(v1[r][c + 5] + v2[r][c + 5]);
                    elements.push_back(v1[r][c + 6] + v2[r][c + 6]);
                    elements.push_back(v1[r][c + 7] + v2[r][c + 7]);
                    elements.push_back(v1[r][c + 8] + v2[r][c + 8]);
                    elements.push_back(v1[r][c + 9] + v2[r][c + 9]);
                }
                item.push_back(elements);
            }
            return item;
        }

        const vector <vector <double> > etaADD(const vector <vector <double> > &v1, const vector <vector <double> > &v2)
        {
            vector <vector <double> > item;

            for (int r = 0; r < v1.size(); r ++)
            {
                vector <double> elements;
                for (int c = 0; c < v1[0].size()-6; c += 7)
                {
                    elements.push_back(v1[r][c] + v2[r][c]);
                    elements.push_back(v1[r][c + 1] + v2[r][c + 1]);
                    elements.push_back(v1[r][c + 2] + v2[r][c + 2]);
                    elements.push_back(v1[r][c + 3] + v2[r][c + 3]);
                    elements.push_back(v1[r][c + 4] + v2[r][c + 4]);
                    elements.push_back(v1[r][c + 5] + v2[r][c + 5]);
                    elements.push_back(v1[r][c + 6] + v2[r][c + 6]);
                }
                item.push_back(elements);
            }
            return item;
        }

        const vector <vector <double> > pentaADD(const vector <vector <double> > &v1, const vector <vector <double> > &v2)
        {
            vector <vector <double> > item;

            for (int r = 0; r < v1.size(); r ++)
            {
                vector <double> elements;
                for (int c = 0; c < v1[0].size()-4; c += 5)
                {
                    elements.push_back(v1[r][c] + v2[r][c]);
                    elements.push_back(v1[r][c + 1] + v2[r][c + 1]);
                    elements.push_back(v1[r][c + 2] + v2[r][c + 2]);
                    elements.push_back(v1[r][c + 3] + v2[r][c + 3]);
                    elements.push_back(v1[r][c + 4] + v2[r][c + 4]);
                }
                item.push_back(elements);
            }
            return item;
        }

        const vector <vector <double> > quadraADD(const vector <vector <double> > &v1, const vector <vector <double> > &v2)
        {
            vector <vector <double> > item;

            for (int r = 0; r < v1.size(); r ++)
            {
                vector <double> elements;
                for (int c = 0; c < v1[0].size()-3; c += 4)
                {
                    elements.push_back(v1[r][c] + v2[r][c]);
                    elements.push_back(v1[r][c + 1] + v2[r][c + 1]);
                    elements.push_back(v1[r][c + 2] + v2[r][c + 2]);
                    elements.push_back(v1[r][c + 3] + v2[r][c + 3]);
                }
                item.push_back(elements);
            }
            return item;
        }

        const vector <vector <double> > doubleADD(const vector <vector <double> > &v1, const vector <vector <double> > &v2)
        {
            vector <vector <double> > item;

            for (int r = 0; r < v1.size(); r ++)
            {
                vector <double> elements;
                for (int c = 0; c < v1[0].size()-1; c += 2)
                {
                    elements.push_back(v1[r][c] + v2[r][c]);
                    elements.push_back(v1[r][c + 1] + v2[r][c + 1]);
                }
                item.push_back(elements);
            }
            return item;
        }

        const vector <vector <double> > singleADD(const vector <vector <double> > &v1, const vector <vector <double> > &v2)
        {
            vector <vector <double> > item;

            for (int r = 0; r < v1.size(); r ++)
            {
                vector <double> elements;
                for (int c = 0; c < v1[0].size(); c ++)
                {
                    elements.push_back(v1[r][c] + v2[r][c]);
                }
                item.push_back(elements);
            }
            return item;
        }
    
    public:    
        const vector <vector <double> > matrix_add(const vector <vector <double> > &v1, const vector <vector <double> > &v2)
        {
            assert(v1.size() == v2.size() && v1[0].size() == v2[0].size() && "The size of the vectors do not match!");
            
            int target_size = v1[0].size();

            if (target_size > 10)
            {
                if (target_size % 10 == 0)
                {
                    return dekaADD(v1, v2);
                }
                else if (target_size % 7 == 0)
                {
                    return etaADD(v1, v2);
                }
                else if (target_size % 5 == 0)
                {
                    return pentaADD(v1, v2);
                }
                else if (target_size % 4 == 0)
                {
                    return quadraADD(v1, v2);
                }
                else if (target_size % 2 == 0)
                {
                    return doubleADD(v1, v2);
                }
                else 
                {
                    return singleADD(v1, v2);
                }
            }
            else 
            {
                return singleADD(v1, v2);
            }
        }
};

#endif