bool ValidateC4Graph(Graph& graph) {
    int i = 0;
    for (const auto& node_pair : graph.nodes) {
        if(i % 100 == 0)
            cout << i << endl;
        i++;
        const Node& node = node_pair.second;
        const C4Board& board = node.data;

        C4Board copy = board;

        // Get children hashes and check validity
        unordered_set<double> expected_children_hashes = copy.get_children_hashes();

        if (board.is_reds_turn()) {

            // Case 1: Red-to-move node has exactly one child
            if (node.neighbors.size() == 1) {
                const Edge& child_edge = *node.neighbors.begin();
                double child_hash = child_edge.to;

                if (!graph.node_exists(child_hash)) {
                    cout << "Invalid: Red-to-move node has a non-existent child." << endl;
                    return false;
                }

                const Node<C4Board>& child_node = graph.nodes.at(child_hash);
                const C4Board& child_board = *child_node.data;

                if (child_board.is_reds_turn()) {
                    cout << "Invalid: Red-to-move node has a red-to-move child." << endl;
                    return false;
                }
            }

            // Case 2: Red-to-move node has no children and must have a valid steady state
            else if (node.neighbors.empty()) {
                shared_ptr<SteadyState> ss = find_cached_steady_state(board);
                if (ss == nullptr){
                    cout << "Invalid: Red-to-move node has no nodes and no steadystate" << endl;
                    return false;
                }
                if (!ss->validate(board)){
                    cout << "Invalid: Red-to-move node's steady state failed validation" << endl;
                    return false;
                }
            }

            else {
                cout << "Invalid: Red-to-move node must have exactly one child or no children with a steady state." << endl;
                return false;
            }
        }

        else { // Yellow-to-move
            // Case 3: Yellow-to-move node must have children corresponding to all legal moves
            unordered_set<double> actual_children_hashes;
            for (const auto& edge : node.neighbors) {
                actual_children_hashes.insert(edge.to);
            }

            if (actual_children_hashes != expected_children_hashes) {
                cout << "Invalid: Yellow-to-move node does not have all legal children." << endl;
                return false;
            }

            // Validate the board state: Red must win under perfect play
            int work;
            if (copy.who_is_winning(work, false) != RED) {
                cout << "Invalid: Yellow-to-move node is not Red-to-win under perfect play." << endl;
                return false;
            }
        }
    }

    // If all checks pass, the graph is valid
    cout << "Graph is a valid Connect 4 weak solution tree." << endl;
    return true;
}
