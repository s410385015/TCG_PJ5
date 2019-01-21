/**
 * Framework for 2048 & 2048-like Games (C++ 11)
 * use 'g++ -std=c++11 -O3 -g -o 2048 2048.cpp' to compile the source
 *
 * Author: Hung Guei (moporgic)
 *         Computer Games and Intelligence (CGI) Lab, NCTU, Taiwan
 *         http://www.aigames.nctu.edu.tw
 */

#pragma once
#include <iostream>
#include <vector>
#include <array>
#include <utility>

class weight {
public:
	weight() {}
	weight(size_t len,const std::vector<int>& p) : value(len) {
		setWeight(p);
	}
	weight(weight&& f) : value(std::move(f.value)) {}
	weight(const weight& f) = default;

	weight& operator =(const weight& f) = default;
	float& operator[] (size_t i) { return value[i]; }
	const float& operator[] (size_t i) const { return value[i]; }
	size_t size() const { return value.size(); }

public:
	friend std::ostream& operator <<(std::ostream& out, const weight& w) {
		auto& value = w.value;
		uint64_t size = value.size();
		out.write(reinterpret_cast<const char*>(&size), sizeof(uint64_t));
		out.write(reinterpret_cast<const char*>(value.data()), sizeof(float) * size);
		return out;
	}
	friend std::istream& operator >>(std::istream& in, weight& w) {
		auto& value = w.value;
		uint64_t size = 0;
		in.read(reinterpret_cast<char*>(&size), sizeof(uint64_t));
		value.resize(size);
		in.read(reinterpret_cast<char*>(value.data()), sizeof(float) * size);
		return in;
	}

	float estimate(const board& b)const{
		float value=0;
		for (int i=0;i<iso_num;i++){
			
			size_t index=indexof(isomorphic[i],b);
			value+=operator[](index);
		}
		return value;
	} 

	float update(const board& b,float u)
	{
		float a=u/iso_num;
		float value=0;

		for (int i = 0; i < iso_num; i++) {
			size_t index = indexof(isomorphic[i], b);
			//std::cout<<index<<std::endl;
			operator[](index) += a;
			value += operator[](index);
		}
		return value;
	}

	size_t indexof(const std::vector<int>& patt, const board& b) const {
		size_t index = 0;
	
		for (size_t i = 0; i < patt.size(); i++){	
			index |= b(patt[i]) << (4 * i);
		
		}
		return index;
	}


	uint64_t mirror(uint64_t idx)
	{
		idx = ((idx & 0x000f000f000f000fULL) << 12) | ((idx & 0x00f000f000f000f0ULL) << 4)
		    | ((idx & 0x0f000f000f000f00ULL) >> 4) | ((idx & 0xf000f000f000f000ULL) >> 12);
		return idx;
	}

	uint64_t transpose(uint64_t idx) {
		idx = (idx & 0xf0f00f0ff0f00f0fULL) | ((idx & 0x0000f0f00000f0f0ULL) << 12) | ((idx & 0x0f0f00000f0f0000ULL) >> 12);
		idx = (idx & 0xff00ff0000ff00ffULL) | ((idx & 0x00000000ff00ff00ULL) << 24) | ((idx & 0x00ff00ff00000000ULL) >> 24);
		return idx;
	}

	uint64_t flip(uint64_t idx) {
		idx = ((idx & 0x000000000000ffffULL) << 48) | ((idx & 0x00000000ffff0000ULL) << 16)
		    | ((idx & 0x0000ffff00000000ULL) >> 16) | ((idx & 0xffff000000000000ULL) >> 48);
		return idx;
	}

	uint64_t rotate(uint64_t idx,int r=1)
	{
		switch (((r % 4) + 4) % 4) {
		default:
		case 0: return idx;
		case 1: 
			return mirror(transpose(idx));
			break;
		case 2: 
			return flip(transpose(idx));
			break;
		case 3:
			return flip(mirror(idx)); 
			break;
		}
	}

	int at(uint64_t idx,int i)
	{
		return (idx >> (i << 2)) & 0x0f;
	}


	void setWeight(const std::vector<int>& p)
	{
		if(p.empty()) return;

		for (int i = 0; i < 8; i++) {
			uint64_t idx= 0xfedcba9876543210ull;
			if (i >= 4) idx=mirror(idx);
			idx=rotate(idx,i);
			for (int t : p) {
				isomorphic[i].push_back(at(idx,t));
			}
		}
	}

protected:
	std::vector<float> value;
	std::array<std::vector<int>, 8> isomorphic;
	int iso_num=8;
};


class state{
		private:
			board before;
			board after;
			float value;
			int reward;
			int hint;
			int last;
		public:
			state(){}
			board GetAfter() const { return after; }
			board GetBefore() const { return before; }
			float GetValue() const { return value; }
			int GetReward() const { return reward; }
			int GetHint() const{return hint;}
			int GetLast() const{return last;}

			void SetBefore(const board& b) { before = b; }
			void SetAfter(const board& b) { after = b; }
			void SetValue(float v) { value = v; }
			void SetReward(int r) { reward = r; }
			void SetHint(int h){hint=h;}
			void SetLast(int l){last=l;}

};