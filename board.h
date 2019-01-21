#pragma once
#include <array>
#include <iostream>
#include <iomanip>

/**
 * array-based board for 2048
 *
 * index (1-d form):
 *  (0)  (1)  (2)  (3)
 *  (4)  (5)  (6)  (7)
 *  (8)  (9) (10) (11)
 * (12) (13) (14) (15)
 *
 */
class board {
public:
	typedef uint32_t cell;
	typedef std::array<cell, 4> row;
	typedef std::array<row, 4> grid;
	typedef uint64_t data;
	typedef int reward;

public:
	board() : tile(), attr(0),last_move(-1) {}
	board(const grid& b, data v = 0) : tile(b), attr(v) {}
	board(const board& b) = default;
	board& operator =(const board& b) = default;

	operator grid&() { return tile; }
	operator const grid&() const { return tile; }
	row& operator [](unsigned i) { return tile[i]; }
	const row& operator [](unsigned i) const { return tile[i]; }
	cell& operator ()(unsigned i) { return tile[i / 4][i % 4]; }
	const cell& operator ()(unsigned i) const { return tile[i / 4][i % 4]; }

	data info() const { return attr; }
	data info(data dat) { data old = attr; attr = dat; return old; }


public:
	bool operator ==(const board& b) const { return tile == b.tile; }
	bool operator < (const board& b) const { return tile <  b.tile; }
	bool operator !=(const board& b) const { return !(*this == b); }
	bool operator > (const board& b) const { return b < *this; }
	bool operator <=(const board& b) const { return !(b < *this); }
	bool operator >=(const board& b) const { return !(*this < b); }

public:

	/**
	 * place a tile (index value) to the specific position (1-d form index)
	 * return 0 if the action is valid, or -1 if not
	 */
	reward place(unsigned pos, cell tile,cell next) {
		//std::cout<<"!!:"<<pos<<" "<<tile<<" "<<next<<std::endl;
		if (pos >= 16) return -1;
		//if (tile != 1 && tile != 2 && tile != 3) return -1;
		operator()(pos) = tile;
		nextTile=next;
		return 0;
	}

	/**
	 * apply an action to the board
	 * return the reward of the action, or -1 if the action is illegal
	 */
	reward slide(unsigned opcode) {
		last_move=opcode&0b11;

		//std::cout<<last_move<<std::endl;
		switch (opcode & 0b11) {
		case 0: return slide_up();
		case 1: return slide_right();
		case 2: return slide_down();
		case 3: return slide_left();
		default: return -1;
		}
	}

	reward slide_left() {
		board prev = *this;
		reward score = 0;
		for (int r = 0; r < 4; r++) {
			auto& row = tile[r];
			for (int c = 0; c < 3; c++) {
				int tile_1 = row[c];
				int tile_2 = row[c+1];
				if(!tile_1)
				{	
					row[c]=tile_2;
					row[c+1]=0;
					continue;
				}

				if( tile_1!=0 && tile_2!=0 && tile_1+tile_2==3 )
				{
					row[c]=3;
					row[c+1]=0;
					score+=3;
					continue;
				}

				if(tile_1==tile_2&& tile_1>2 && tile_1<15)
				{
					row[c]=tile_1+1;
					row[c+1]=0;
					score+=(3<<(tile_1-2));
					continue;
				}
				
			}
			
		}
		return (*this != prev) ? score : -1;
	}
	reward slide_right() {
		reflect_horizontal();
		reward score = slide_left();
		reflect_horizontal();
		return score;
	}
	reward slide_up() {
		rotate_right();
		reward score = slide_right();
		rotate_left();
		return score;
	}
	reward slide_down() {
		rotate_right();
		reward score = slide_left();
		rotate_left();
		return score;
	}

	void transpose() {
		for (int r = 0; r < 4; r++) {
			for (int c = r + 1; c < 4; c++) {
				std::swap(tile[r][c], tile[c][r]);
			}
		}
	}

	void reflect_horizontal() {
		for (int r = 0; r < 4; r++) {
			std::swap(tile[r][0], tile[r][3]);
			std::swap(tile[r][1], tile[r][2]);
		}
	}

	void reflect_vertical() {
		for (int c = 0; c < 4; c++) {
			std::swap(tile[0][c], tile[3][c]);
			std::swap(tile[1][c], tile[2][c]);
		}
	}

	/**
	 * rotate the board clockwise by given times
	 */
	void rotate(int r = 1) {
		switch (((r % 4) + 4) % 4) {
		default:
		case 0: break;
		case 1: rotate_right(); break;
		case 2: reverse(); break;
		case 3: rotate_left(); break;
		}
	}
	cell GetMaxTile()const
	{
		cell t=0;
		for(int i=0;i<4;i++)
			for(int j=0;j<4;j++)
				if(tile[i][j]>t)
					t=tile[i][j];
		return t;
	}
	void rotate_right() { transpose(); reflect_horizontal(); } // clockwise
	void rotate_left() { transpose(); reflect_vertical(); } // counterclockwise
	void reverse() { reflect_horizontal(); reflect_vertical(); }
	reward getLastMove()const {return last_move;}
	reward getNextTile()const {return nextTile;}
public:
	friend std::ostream& operator <<(std::ostream& out, const board& b) {
		const char* idx = "012";
		out << "+------------------------+" << std::endl;
		for (auto& row : b.tile) {
			out << "|" << std::dec;
			for (auto t : row) out << std::setw(6) <<((t>2)? ((3<<(t-3))):(idx[t]-'0'));
			out << "|" << std::endl;
		}
		out << "+------------------------+" << std::endl;
		return out;
	}

private:
	grid tile;
	data attr;
	reward last_move;
	reward nextTile;
};
