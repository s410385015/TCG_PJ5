#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include "board.h"
#include "action.h"
#include "weight.h"
#include <fstream>
#include <limits.h>
#include "TD.h"
#include <deque>

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }
	virtual void InitEpisode(){}
public:
	virtual std::string property(const std::string& key) const { return meta.at(key); }
	virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }

protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
};

class random_agent : public agent {
public:
	random_agent(const std::string& args = "") : agent(args) {
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

/**
 * base agent for agents with weight tables
 */
 /*
class weight_agent : public agent {
public:
	weight_agent(const std::string& args = "") : agent(args) {
		if (meta.find("init") != meta.end()) // pass init=... to initialize the weight
			init_weights(meta["init"]);
		if (meta.find("load") != meta.end()) // pass load=... to load from a specific file
			load_weights(meta["load"]);
	}
	virtual ~weight_agent() {
		if (meta.find("save") != meta.end()) // pass save=... to save to a specific file
			save_weights(meta["save"]);
	}

protected:
	virtual void init_weights(const std::string& info) {
		net.emplace_back(65536); // create an empty weight table with size 65536
		net.emplace_back(65536); // create an empty weight table with size 65536
		// now net.size() == 2; net[0].size() == 65536; net[1].size() == 65536
	}
	virtual void load_weights(const std::string& path) {
		std::ifstream in(path, std::ios::in | std::ios::binary);
		if (!in.is_open()) std::exit(-1);
		uint32_t size;
		in.read(reinterpret_cast<char*>(&size), sizeof(size));
		net.resize(size);
		for (weight& w : net) in >> w;
		in.close();
	}
	virtual void save_weights(const std::string& path) {
		std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
		if (!out.is_open()) std::exit(-1);
		uint32_t size = net.size();
		out.write(reinterpret_cast<char*>(&size), sizeof(size));
		for (weight& w : net) out << w;
		out.close();
	}

protected:
	std::vector<weight> net;
};
*/
/**
 * base agent for agents with a learning rate
 */
class learning_agent : public agent {
public:
	learning_agent(const std::string& args = "") : agent(args), alpha(0.1f) {
		if (meta.find("alpha") != meta.end())
			alpha = float(meta["alpha"]);
	}
	virtual ~learning_agent() {}

protected:
	float alpha;
};

/**
 * random environment
 * add a new random tile to an empty cell
 * 2-tile: 90%
 * 4-tile: 10%
 */
class rndenv : public random_agent {
public:
	rndenv(const std::string& args = "") : random_agent("name=random role=environment " + args),
		opcode({  0,1, 2, 3 }),
		space({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }), 
		space_each_op({12,13,14,15,0,4,8,12,0,1,2,3,3,7,11,15})
		{
			td.add(new weight(1<<6*sizeof(float),{0, 1, 2, 3, 4, 5}));
			td.add(new weight(1<<6*sizeof(float),{4, 5, 6, 7, 8, 9}));
			td.add(new weight(1<<6*sizeof(float),{0, 1, 2, 4, 5, 6 }));
			td.add(new weight(1<<6*sizeof(float),{4, 5, 6, 8, 9, 10 }));
			TD_load();
			
		}

	virtual void InitEpisode(){
		ResetBags();
	}
	virtual action take_action(const board& after) {
	
		
		total++;
		std::shuffle(space.begin(), space.end(), engine);
		if(bags.size() <3) FillIn();
		
	
		board::cell tile=GenerateTile(after);
		board::cell hintTile=bags.front();

		if(bonusFlag)
			hintTile=4;

		int last_move=after.getLastMove();
		//std::cout<<last_move<<std::endl;
		
		if(last_move==-1){
			for (int pos : space) {
				if (after(pos) != 0) continue;	
						
				return action::place(pos, tile,hintTile);
			}
		}else
		{
			//std::shuffle(space_each_op[last_move].begin(), space_each_op[last_move].end(), engine);
			action act=action();
			board:: reward _min=INT_MAX;
			for(int pos : space_each_op[last_move])
			{
				//std::cout<<pos<<" ";
				if (after(pos) != 0) continue;			
				
				board nextBoard(after);
				nextBoard.place(pos,tile,hintTile);
				
				board::reward _max=INT_MIN;
				for (int op : opcode) {
					board _after(nextBoard);
					board::reward reward = _after.slide(op);

					if (reward == -1)
						continue;
					board:: reward new_reward=reward+td.estimate(_after,bags.front()-1)+Search(_after,hintTile,1);


					if(new_reward>_max)
						_max=new_reward;
					
				}

				if(_max<_min){
					_min=_max;
					act=action::place(pos,tile,hintTile);
				}
			
			}


			return act;
			//return action::place(pos, tile,bags.front());
		}
		return action();
	}

	board:: reward Search(board after,board::cell tile,int i)
	{
	
		int last_move=after.getLastMove();
		board::cell hintTile=bags[i];
		board:: reward _min=INT_MAX;
		for(int pos : space_each_op[last_move])
		{
			//std::cout<<pos<<" ";
			if (after(pos) != 0) continue;			
			
			board nextBoard(after);
			nextBoard.place(pos,tile,hintTile);
			
			board::reward _max=INT_MIN;
			for (int op : opcode) {
				board _after(nextBoard);
				board::reward reward = _after.slide(op);

				if (reward == -1)
					continue;
				board:: reward new_reward=reward+td.estimate(_after,hintTile-1);


				if(new_reward>_max)
					_max=new_reward;
				
			}

			if(_max<_min){
				_min=_max;
		
			}
		}
		return _min;
	}
	

	board::cell GenerateTile(const board& b)
	{

		if(bonusFlag)
		{
			board::cell max_tile=b.GetMaxTile();
			bonusFlag=false;
			//return 4;
			return 4+rand()%(max_tile-7+1);
		}

		board::cell max_tile=b.GetMaxTile();
		
		board::cell tile=-1;
		if(max_tile>=7){

		
			if((rand()%21==0)&&((bonus+1)/total)<(1.0/21.0))
			{
				bonusFlag=true;
				bonus++;
			}
			
			
		}
		
		tile= bags.front();
		bags.pop_front();
			//std::cout<<"?"<<std::endl;
		
		//std::cout<<max_tile<<std::endl;
		//std::cout<<tile<<std::endl;
		return tile;
	}



	void FillIn()
	{	
		std::array<int,12> tmp{1,1,1,1,2,2,2,2,3,3,3,3};
		std::shuffle(tmp.begin(),tmp.end(),engine);
		for(int i=0;i<12;i++)
			bags.push_back(tmp[i]);
			
	}
	void ResetBags()
	{
		bags.clear();
		FillIn();
		total=bonus=0;
		bonusFlag=false;
	}
	void TD_load()
	{
		td.Load();
	}
private:
	std::array<int, 4> opcode;
	std::array<int, 16> space;
	std::array<std::array<int,4>,4> space_each_op;
	std::deque<int> bags;
	float total;
	float bonus;
	TD td;
	bool bonusFlag=false;
	
};

/**
 * dummy player
 * select a legal action randomly
 */
class player : public random_agent {
public:
	player(const std::string& args = "") : random_agent("name=dummy role=player " + args),
		opcode({  0,1, 2, 3 }),
		space_each_op({12,13,14,15,0,4,8,12,0,1,2,3,3,7,11,15}) {
			path.reserve(20000);
			td.add(new weight(1<<6*sizeof(float),{0, 1, 2, 3, 4, 5}));
			td.add(new weight(1<<6*sizeof(float),{4, 5, 6, 7, 8, 9}));
			td.add(new weight(1<<6*sizeof(float),{0, 1, 2, 4, 5, 6 }));
			td.add(new weight(1<<6*sizeof(float),{4, 5, 6, 8, 9, 10 }));
			TD_load();
		}

	virtual action take_action(const board& before) {

		return SelectMove(before);
	}

	action SelectMove(const board& before)
	{
		
		board:: reward _max=INT_MIN;
		action act=action();
		
	
		for (int op : opcode) {
			board after(before);
			board::reward reward = after.slide(op);
			//std::cout<<op<<" "<<reward<<std::endl;
			if (reward == -1)
				continue;

			board:: reward new_reward=reward;
			
			if(before.getNextTile()<=3)
				new_reward+=td.estimate(after,before.getNextTile()-1);
			
			if(new_reward>_max)
			{
				_max=new_reward;
				act=action::slide(op);
			}
		}

	
		return act;
	}


	
	void TD_load()
	{
		td.Load();
	}
private:
	std::array<int, 4> opcode;
	std::array<std::array<int,4>,4> space_each_op;
	std::vector<state> path;
	TD td;
};
