#include <chrono>

bool ValidateC4Graph(Graph& graph) {
    auto start_time = std::chrono::high_resolution_clock::now();

    int i = 0;
    bool valid = true;
    for (const auto& node_pair : graph.nodes) {
        if(i % 100 == 0)
            cout << i << endl;
        i++;
        const Node& node = node_pair.second;
        const C4Board& board = *dynamic_cast<C4Board*>(node.data);

        C4Board copy = board;

        // Get children hashes and check validity
        unordered_set<double> expected_children_hashes = copy.get_children_hashes();

        auto children = node.neighbors;
        // Ignore neighbors which are parents by deleting neighbors whose representation is one character longer than the current node's representation
        for (auto it = children.begin(); it != children.end();) {
            const Node& neighbor_node = graph.nodes.at(it->to);
            if (neighbor_node.data->representation.size() + 1 == node.data->representation.size()) {
                it = children.erase(it); // Remove this neighbor
            } else {
                ++it; // Move to the next neighbor
            }
        }

        if (board.is_reds_turn()) {
            // Case 1: Red-to-move node has exactly one child
            if (children.size() == 1) {
                const Edge& child_edge = *children.begin();
                double child_hash = child_edge.to;

                if (!graph.node_exists(child_hash)) {
                    cout << "Invalid: Red-to-move node has a non-existent child." << endl;
                    cout << node.data->representation << endl << endl;
                    valid = false;
                }

                const Node& child_node = graph.nodes.at(child_hash);
                const C4Board& child_board = *dynamic_cast<C4Board*>(child_node.data);

                if (child_board.is_reds_turn()) {
                    cout << "Invalid: Red-to-move node has a red-to-move child." << endl;
                    cout << node.data->representation << endl << endl;
                    valid = false;
                }
            }

            // Case 2: Red-to-move node has no children and must have a valid steady state
            else if (children.empty()) {
                shared_ptr<SteadyState> ss = find_cached_steady_state(board);
                if (ss == nullptr){
                    cout << "Invalid: Red-to-move node has no nodes and no steadystate" << endl;
                    cout << node.data->representation << endl << endl;
                    valid = false;
                }
                if (!ss->validate(board)){
                    cout << "Invalid: Red-to-move node's steady state failed validation" << endl;
                    cout << node.data->representation << endl << endl;
                    valid = false;
                }
            }

            else {
                cout << "Invalid: Red-to-move node must have exactly one child or no children with a steady state." << endl;
                cout << "Number of children: " << children.size() << endl;
                cout << "Steady state: " << (find_cached_steady_state(board) != nullptr ? "exists" : "does not exist") << endl;
                cout << node.data->representation << endl << endl;
                valid = false;
            }
        }

        else { // Yellow-to-move
            // Case 3: Yellow-to-move node must have children corresponding to all legal moves
            unordered_set<double> actual_children_hashes;
            for (const auto& edge : children) {
                actual_children_hashes.insert(edge.to);
            }

            // Check that all expected children are present
            for (const auto& hash : expected_children_hashes) {
                if (actual_children_hashes.find(hash) == actual_children_hashes.end()) {
                    cout << "Invalid: Yellow-to-move node has missing child: " << hash << endl;
                    cout << node.data->representation << endl << endl;
                    valid = false;
                }
            }
            for (const auto& hash : actual_children_hashes) {
                if (expected_children_hashes.find(hash) == expected_children_hashes.end()) {
                    cout << "Invalid: Yellow-to-move node has unexpected child: " << hash << endl;
                    cout << node.data->representation << endl << endl;
                    valid = false;
                }
            }

            // Validate the board state: Red must win under perfect play
            int work;
            if (copy.who_is_winning(work, false) != RED) {
                cout << "Invalid: Yellow-to-move node is not Red-to-win under perfect play." << endl;
                cout << node.data->representation << endl << endl;
                valid = false;
            }
        }
    }

    // If all checks pass, the graph is valid
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    cout << "Validation result: " << (valid ? "Valid" : "Invalid") << endl;
    cout << "Validation took " << elapsed_seconds.count() << " seconds." << endl;
    return valid;
}
