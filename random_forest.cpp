#include <iostream>
#include <vector>
#include <random>
#include <map>
#include <algorithm>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <string>
#include <omp.h>
#include <chrono>

#define SEED 11
#define NUM_THREADS 4
#define NUM_TREES 32

// Estrutura para manter os dados de forma organizada
struct Dataset {
    std::vector<std::vector<double>> features;
    std::vector<int> labels;
};

// =================================================================================
// FUNÇÃO PARA CARREGAR E ENCODAR O CSV
// =================================================================================
Dataset load_csv_and_encode(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Não foi possível abrir o arquivo: " + filepath);
    }

    Dataset data;
    std::string line;
    
    // Pula a linha do cabeçalho
    std::getline(file, line); 

    // Mapeamento dinâmico para cada coluna categórica
    std::vector<std::map<std::string, int>> categorical_maps;
    int num_columns = 0;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string cell;
        std::vector<double> feature_row;
        int current_col = 0;

        // Inicializa os mapas na primeira linha de dados
        if (num_columns == 0) {
            std::stringstream temp_ss(line);
            while (std::getline(temp_ss, cell, ',')) {
                num_columns++;
            }
            categorical_maps.resize(num_columns);
        }

        while (std::getline(ss, cell, ',')) {
            // A última coluna é o rótulo (label)
            if (current_col == num_columns - 1) {
                if (categorical_maps[current_col].find(cell) == categorical_maps[current_col].end()) {
                    categorical_maps[current_col][cell] = categorical_maps[current_col].size();
                }
                data.labels.push_back(categorical_maps[current_col][cell]);
            } 
            // As outras colunas são features
            else {
                // Tenta converter para double, se falhar, trata como categórico
                try {
                    feature_row.push_back(std::stod(cell));
                } catch (const std::invalid_argument&) {
                    if (categorical_maps[current_col].find(cell) == categorical_maps[current_col].end()) {
                        categorical_maps[current_col][cell] = categorical_maps[current_col].size();
                    }
                    feature_row.push_back(static_cast<double>(categorical_maps[current_col][cell]));
                }
            }
            current_col++;
        }
        data.features.push_back(feature_row);
    }
    
    std::cout << "CSV carregado com sucesso!" << std::endl;
    std::cout << "Número de amostras: " << data.features.size() << std::endl;
    std::cout << "Número de features: " << (data.features.empty() ? 0 : data.features[0].size()) << std::endl;
    
    file.close();
    return data;
}

// =================================================================================
// FUNÇÃO PARA DIVIDIR O DATASET EM TREINO E TESTE
// =================================================================================
std::pair<Dataset, Dataset> train_test_split(const Dataset& full_data, double test_size = 0.2) {
    if (full_data.features.empty()) {
        throw std::runtime_error("Dataset de entrada para split está vazio.");
    }

    Dataset train_data, test_data;
    
    // 1. Criar um vetor de índices de 0 a N-1
    std::vector<size_t> indices(full_data.features.size());
    std::iota(indices.begin(), indices.end(), 0);

    // 2. Embaralhar os índices aleatoriamente
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // 3. Determinar o ponto de corte
    size_t split_point = static_cast<size_t>(full_data.features.size() * (1.0 - test_size));

    // 4. Popular o dataset de treino
    for (size_t i = 0; i < split_point; ++i) {
        train_data.features.push_back(full_data.features[indices[i]]);
        train_data.labels.push_back(full_data.labels[indices[i]]);
    }

    // 5. Popular o dataset de teste
    for (size_t i = split_point; i < indices.size(); ++i) {
        test_data.features.push_back(full_data.features[indices[i]]);
        test_data.labels.push_back(full_data.labels[indices[i]]);
    }

    std::cout << "\nDataset dividido:" << std::endl;
    std::cout << "Amostras de treino: " << train_data.features.size() << std::endl;
    std::cout << "Amostras de teste:  " << test_data.features.size() << std::endl;

    return {train_data, test_data};
}

// =================================================================================
// CLASSE DA ÁRVORE DE DECISÃO
// =================================================================================
class DecisionTree {
private:
    // Nó da árvore
    struct Node {
        bool is_leaf = false;
        int prediction = -1; // Predição se for folha

        int feature_index = -1;
        double split_value = 0.0;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
    };

    std::unique_ptr<Node> root;
    int max_depth;
    int min_samples_split;
    int num_features_subset;

    // Calcula a impureza de Gini para um conjunto de rótulos
    double calculate_gini(const std::vector<int>& labels) {
        if (labels.empty()) return 0.0;

        std::map<int, int> class_counts;
        for (int label : labels) {
            class_counts[label]++;
        }

        double impurity = 1.0;
        for (const auto& pair : class_counts) {
            double prob = static_cast<double>(pair.second) / labels.size();
            impurity -= prob * prob;
        }
        return impurity;
    }

    // Encontra a melhor divisão (feature e valor) para um conjunto de dados
    std::pair<int, double> find_best_split(const Dataset& data, const std::vector<int>& indices) {
        double best_gini = 1.0;
        int best_feature = -1;
        double best_value = 0.0;

        double current_gini = calculate_gini(data.labels);

        // Seleciona um subconjunto aleatório de features
        std::vector<int> feature_indices(data.features[0].size());
        std::iota(feature_indices.begin(), feature_indices.end(), 0);
        
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(feature_indices.begin(), feature_indices.end(), g);
        
        int features_to_check = (num_features_subset == -1) ? feature_indices.size() : num_features_subset;

        for (int i = 0; i < features_to_check; ++i) {
            int feature_idx = feature_indices[i];
            
            // Testa valores únicos como pontos de divisão
            std::vector<double> unique_values;
            for (int idx : indices) unique_values.push_back(data.features[idx][feature_idx]);
            std::sort(unique_values.begin(), unique_values.end());
            unique_values.erase(std::unique(unique_values.begin(), unique_values.end()), unique_values.end());

            for (double value : unique_values) {
                std::vector<int> left_labels, right_labels;
                for (int idx : indices) {
                    if (data.features[idx][feature_idx] < value) {
                        left_labels.push_back(data.labels[idx]);
                    } else {
                        right_labels.push_back(data.labels[idx]);
                    }
                }

                if (left_labels.empty() || right_labels.empty()) continue;

                double p_left = static_cast<double>(left_labels.size()) / indices.size();
                double p_right = static_cast<double>(right_labels.size()) / indices.size();
                double gini = p_left * calculate_gini(left_labels) + p_right * calculate_gini(right_labels);

                if (gini < best_gini) {
                    best_gini = gini;
                    best_feature = feature_idx;
                    best_value = value;
                }
            }
        }
        return {best_feature, best_value};
    }

    // Retorna a classe mais comum em um conjunto de rótulos
    int most_common_label(const std::vector<int>& labels) {
        std::map<int, int> counts;
        for (int label : labels) counts[label]++;
        return std::max_element(counts.begin(), counts.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; })->first;
    }

    // Constrói a árvore recursivamente
    std::unique_ptr<Node> build_tree(const Dataset& data, const std::vector<int>& indices, int depth) {
        auto node = std::make_unique<Node>();
        std::vector<int> current_labels;
        for(int idx : indices) current_labels.push_back(data.labels[idx]);

        // Condições de parada para criar um nó folha
        if (depth >= max_depth || indices.size() < min_samples_split || std::all_of(current_labels.begin(), current_labels.end(), [&](int l){ return l == current_labels[0]; })) {
            node->is_leaf = true;
            node->prediction = most_common_label(current_labels);
            return node;
        }

        auto [feature, value] = find_best_split(data, indices);

        if (feature == -1) { // Não encontrou uma divisão útil
            node->is_leaf = true;
            node->prediction = most_common_label(current_labels);
            return node;
        }
        
        node->feature_index = feature;
        node->split_value = value;
        
        std::vector<int> left_indices, right_indices;
        for (int idx : indices) {
            if (data.features[idx][feature] < value) {
                left_indices.push_back(idx);
            } else {
                right_indices.push_back(idx);
            }
        }

        node->left = build_tree(data, left_indices, depth + 1);
        node->right = build_tree(data, right_indices, depth + 1);

        return node;
    }

    int predict_single(const std::vector<double>& features, Node* node) {
        if (node->is_leaf) {
            return node->prediction;
        }
        if (features[node->feature_index] < node->split_value) {
            return predict_single(features, node->left.get());
        } else {
            return predict_single(features, node->right.get());
        }
    }


public:
    DecisionTree(int depth = 5, int min_samples = 2, int num_features = -1)
        : max_depth(depth), min_samples_split(min_samples), num_features_subset(num_features) {}

    void train(const Dataset& data) {
        std::vector<int> all_indices(data.features.size());
        std::iota(all_indices.begin(), all_indices.end(), 0);
        root = build_tree(data, all_indices, 0);
    }
    
    // Sobrecarga para treinar com um subconjunto de dados (usado pela Random Forest)
     void train(const Dataset& data, const std::vector<int>& indices) {
        root = build_tree(data, indices, 0);
    }

    int predict(const std::vector<double>& features) {
        if (!root) throw std::runtime_error("Tree is not trained yet.");
        return predict_single(features, root.get());
    }
};

// =================================================================================
// CLASSE DA RANDOM FOREST
// =================================================================================
class RandomForest {
private:
    int num_trees;
    int max_depth;
    int min_samples_split;
    int num_features_subset;
    std::vector<DecisionTree> trees;

public:
    RandomForest(int n_trees, int depth, int min_samples = 2)
        : num_trees(n_trees), max_depth(depth), min_samples_split(min_samples) {
            trees.resize(n_trees);
        }

    void train(const Dataset& data) {
        int num_features = data.features[0].size();
        // Heurística comum: sqrt(num_features) para classificação
        num_features_subset = static_cast<int>(sqrt(num_features));

        // A DIRETIVA OPENMP PARA PARALELIZAR O LOOP
        omp_set_num_threads(NUM_THREADS);
        printf("Numero de arvores = %d\n", num_trees);
        fflush(stdout);
        #pragma omp parallel for schedule (dynamic)
        for (int i = 0; i < num_trees; ++i) {
                if (i == 0){
                    printf("Iniciando Random Forest com %d threads\n", omp_get_num_threads());
                    fflush(stdout);
            }
            // 1. Bootstrap Sampling (amostragem com reposição)
            std::vector<int> sample_indices;
            
            // Gerador de números aleatórios seguro para threads
            std::mt19937 generator(std::random_device{}() + omp_get_thread_num());
            std::uniform_int_distribution<int> distribution(0, data.features.size() - 1);
            
            for (size_t j = 0; j < data.features.size(); ++j) {
                sample_indices.push_back(distribution(generator));
            }

            // 2. Treina uma árvore com a amostra
            trees[i] = DecisionTree(max_depth, min_samples_split, num_features_subset);
            trees[i].train(data, sample_indices);
            printf("Treinou arvore %d (thread %d)\n", i, omp_get_thread_num());
	        fflush(stdout);
        }
    }

    int predict(const std::vector<double>& features) {
        std::map<int, int> votes;
        for (int i = 0; i < num_trees; ++i) {
            votes[trees[i].predict(features)]++;
        }
        return std::max_element(votes.begin(), votes.end(),
            [](const auto& a, const auto& b){ return a.second < b.second; })->first;
    }
};


// =================================================================================
// FUNÇÃO MAIN - EXEMPLO DE USO
// =================================================================================
int main() {
    try {
        srand(SEED);
        std::string filename = "weatherAUS_reduced_30.csv";
        Dataset full_data = load_csv_and_encode(filename);

        // AQUI FAZEMOS O SPLIT!
        // Usando structured binding do C++17 para desempacotar o par
        auto [train_data, test_data] = train_test_split(full_data, 0.2); // 80% treino, 20% teste

        RandomForest forest(NUM_TREES, 5);
        
        auto start = std::chrono::high_resolution_clock::now();

        // TREINA APENAS COM OS DADOS DE TREINO
        forest.train(train_data);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;

        double elapsed_time_seconds = elapsed_seconds.count();

        // AVALIA APENAS COM OS DADOS DE TESTE
        std::cout << "\nIniciando avaliação no conjunto de teste..." << std::endl;
        if (test_data.features.empty()) {
            std::cout << "Conjunto de teste está vazio. Não há nada para avaliar." << std::endl;
            return 0;
        }

        int correct_predictions = 0;
        for(size_t i = 0; i < test_data.features.size(); ++i) {
            int prediction = forest.predict(test_data.features[i]);
            int real_label = test_data.labels[i];
            
            std::cout << "Amostra de teste " << i << ": Predição=" << prediction 
                      << ", Real=" << real_label;

            if (prediction == real_label) {
                correct_predictions++;
                std::cout << " (Correto)" << std::endl;
            } else {
                std::cout << " (Incorreto)" << std::endl;
            }
	   fflush(stdout);
        }
        
        // Calcula e exibe a acurácia
        double accuracy = static_cast<double>(correct_predictions) / test_data.features.size();
        std::cout << "\n------------------------------------" << std::endl;
        std::cout << "Acurácia no conjunto de teste: " << accuracy * 100.0 << "%" << std::endl;
        std::cout << "------------------------------------" << std::endl;

        printf("Treinamento concluido em %f segundos\n", elapsed_time_seconds);
	fflush(stdout);

    } catch (const std::exception& e) {
        std::cerr << "Erro: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
