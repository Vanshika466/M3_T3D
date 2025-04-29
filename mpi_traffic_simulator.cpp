#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

const int MASTER = 0;
const int TAG_DATA = 1;
const int TAG_DONE = 2;
const int TAG_RESULT = 3;
const int MAX_LINE_LEN = 256;
const int TOP_N = 3;

void send_data_to_workers(const std::vector<std::string>& lines, int world_size) {
    int worker = 1;
    for (const std::string& line : lines) {
        MPI_Send(line.c_str(), line.size() + 1, MPI_CHAR, worker, TAG_DATA, MPI_COMM_WORLD);
        worker++;
        if (worker == world_size) worker = 1;
    }

    for (int i = 1; i < world_size; ++i) {
        MPI_Send("", 1, MPI_CHAR, i, TAG_DONE, MPI_COMM_WORLD); // Empty line = done
    }
}

void collect_and_display_results(int world_size) {
    std::unordered_map<std::string, int> final_map;

    for (int i = 1; i < world_size; ++i) {
        int count;
        MPI_Recv(&count, 1, MPI_INT, i, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int j = 0; j < count; ++j) {
            char light_id[20];
            int car_count;
            MPI_Recv(light_id, 20, MPI_CHAR, i, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&car_count, 1, MPI_INT, i, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            final_map[light_id] += car_count;
        }
    }

    std::vector<std::pair<std::string, int>> vec(final_map.begin(), final_map.end());
    std::sort(vec.begin(), vec.end(), [](auto& a, auto& b) { return b.second < a.second; });

    std::cout << "\n--- Top " << TOP_N << " Congested Traffic Lights ---\n";
    for (int i = 0; i < std::min(TOP_N, (int)vec.size()); ++i) {
        std::cout << vec[i].first << ": " << vec[i].second << " cars\n";
    }
    std::cout << "----------------------------------------\n";
}

void process_worker() {
    std::unordered_map<std::string, int> local_map;

    while (true) {
        char buffer[MAX_LINE_LEN];
        MPI_Status status;
        MPI_Recv(buffer, MAX_LINE_LEN, MPI_CHAR, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == TAG_DONE) break;

        std::istringstream ss(buffer);
        std::string timestamp, light_id, car_count_str;
        getline(ss, timestamp, ',');
        getline(ss, light_id, ',');
        getline(ss, car_count_str, ',');
        int car_count = std::stoi(car_count_str);

        local_map[light_id] += car_count;
    }

    int map_size = local_map.size();
    MPI_Send(&map_size, 1, MPI_INT, MASTER, TAG_RESULT, MPI_COMM_WORLD);

    for (const auto& [light, count] : local_map) {
        MPI_Send(light.c_str(), 20, MPI_CHAR, MASTER, TAG_RESULT, MPI_COMM_WORLD);
        MPI_Send(&count, 1, MPI_INT, MASTER, TAG_RESULT, MPI_COMM_WORLD);
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 2) {
        if (world_rank == MASTER) {
            std::cerr << "Requires at least 2 processes.\n";
        }
        MPI_Finalize();
        return 1;
    }

    if (world_rank == MASTER) {
        std::ifstream file("traffic_data.txt");
        if (!file.is_open()) {
            std::cerr << "Failed to open traffic_data.txt\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        std::vector<std::string> lines;
        std::string line;
        while (getline(file, line)) {
            lines.push_back(line);
        }

        send_data_to_workers(lines, world_size);
        collect_and_display_results(world_size);
    } else {
        process_worker();
    }

    MPI_Finalize();
    return 0;
}
