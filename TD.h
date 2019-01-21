#pragma once
#include<iostream>
#include<string>
#include<vector>
#include <fstream>
#include"weight.h"
#include"board.h"



class TD{
    private:
        std::array<std::vector<weight*>,3> weights;
        int total=3;
    public:
        TD(){}
        ~TD(){}
    
        void add(weight* w){
        
            for(int i=0;i<total;i++){
                weights[i].push_back(w);
            }
        }

        float estimate(const board& b,int hint)const{
            float value=0;
            
            for(weight* w: weights[hint])
                value+=w->estimate(b);
            
            return value;
        }

        float update(const board& b,float u,int hint) const
        {
           
            float a=u/weights[hint].size();
            float value=0;
            for(weight* w:weights[hint]){
       
                value+=w->update(b,a);
              
                //std::cout<<"Q"<<std::endl;
            }
            return value;
        }

   

        void UpdateEpisode(std::vector<state>& path,float alpha=0.01)const
	    {
    
            float exact=0;
            for (path.pop_back(); path.size(); path.pop_back()) {
                state& move = path.back();
                //std::cout<<"?"<<std::endl;
                float error = exact - (move.GetValue() - move.GetReward());
               
                exact = move.GetReward()+update(move.GetAfter(),alpha*error,move.GetHint()-1);
                
               
		    }
           
	    }

        void Save()
        {

            for(int j=0;j<total;j++){
                std::string num=std::to_string(j);
                
                for(int i=0;i<(int)weights[j].size();i++)
                {
                    std::string name=num+"_"+(std::to_string(i))+"w.txt";
                    std::ofstream out(name, std::ios::out | std::ios::binary | std::ios::trunc);
                    out<<(*weights[j][i]);
                }
            }
        }

        void Load()
        {
            for(int j=0;j<total;j++){
                std::string num=std::to_string(j);
                int i=0;
                for(weight* w:weights[j])
                {
                    std::string name=num+"_"+(std::to_string(i++))+"w.txt";
                    std::ifstream in;
                    in.open(name, std::ios::in | std::ios::binary);
                    in>>*w;
                    in.close();
                }
            }
        }
};