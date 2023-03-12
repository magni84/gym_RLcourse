        if b'A' in desc:
            assert b'a' in desc, "Map with A but no a is not possible"
            state_a = np.where(desc.ravel() == b'a')[0][0]
        else:
            state_a = None
    
        if b'B' in desc:
            assert b'b' in desc, "Map with B but no b is not possible"
            state_b = np.where(desc.ravel() == b'b')[0][0]
        else:
            state_b = None

        if map_name == "Example 3.5":
            step_reward = 0
            outside_reward = -1
        else:
            step_reward = -1
            outside_reward = -1
        
        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)

                for a in range(nA):
                    li = self.P[s][a]
                    letter = desc[row, col]

                    if letter == b'G':
                        li.append( (1.0, s, 0, True) )
                    else:
                        newrow, newcol = inc(row, col, a)
                        newstate = to_s(newrow, newcol)
                        newletter = desc[newrow, newcol]

                        if letter == b'A':
                            newstate = state_a
                            reward = 10.0
                        elif letter == b'B':
                            newstate = state_b
                            reward = 5.0
                        elif newstate == s:
                            reward = outside_reward
                        else:
                            reward = step_reward

                        if letter == b'2':
                            newrow = max(newrow-2, 0)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                        elif letter == b'1':
                            newrow = max(newrow-1, 0)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]

                        terminated = bytes(newletter) in b"G"

                        li.append( (1.0, newstate, reward, terminated) )


WEST = 0
SOUTH = 1
EAST = 2
NORTH = 3
NW = 4
NE = 5
SW = 6
SE = 7
