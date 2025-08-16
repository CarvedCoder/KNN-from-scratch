#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <ios>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <iostream>
#include <random>

class DataPoint{
private:

    std::vector<double> features;
    int label;

public:
        
    DataPoint(const std::vector<double> f , int l) : features(f) , label(l){}

    void Display(){
        std::cout << "Features:"; 
        for(auto &feature:features){
            std::cout << feature << " ";
        }
        
        std::cout << "\n" << "Label:" << label << "\n";
    }

    const std::vector<double>& GetFeatures()const{
        return features;
    } 
    
    int GetLabel()const{
        return label;
    }
};

// class KNN{
// private:
//     std::vector<DataPoint> dataPoints;
//     int K_points;
// };

std::vector<DataPoint> ReadCSV(const std::string&filename){
    std::ifstream file(filename);
    std::vector<DataPoint> data;
    std::string line;
    std::getline(file,line); 
    while(std::getline(file,line)){
        size_t start = 0;
        size_t commapos;
        std::vector<double> features;
        int label;
        while((commapos = line.find(',',start))!= std::string::npos){
            std::string feature = line.substr(start,commapos-start);
            features.push_back(std::stod(feature));
            start = commapos + 1;
        }
        std::string last_val = line.substr(start);
        label = std::stoi(last_val);   
        
        
        data.emplace_back(features,label);

    }
    return data;
              
}

void WriteCSV(const std::vector<DataPoint>&queryPoints,const std::vector<std::vector<std::pair<double, DataPoint>>>& kNearest,const int K_points,const std::vector<DataPoint>&Dataset){
    
    std::ofstream out("KNN.csv");
    if (!out) {
        std::cerr << "Failed to open KNN.csv for writing\n";
        return;
    }
    out << std::fixed << std::setprecision(8);
    size_t feature_count = Dataset[0].GetFeatures().size();

    for(size_t i = 0; i < feature_count;i++){
        out << "feature" << i;
        if(i < feature_count-1) out << ',';
    }
    out << ",label,type,query_id,distance\n";

    for(size_t i =0;i<queryPoints.size();i++){
        const auto& queryPoint = queryPoints[i].GetFeatures();
        for(size_t j =0;j < queryPoint.size();j++ ){
            out << queryPoint[j];
            if(j < queryPoint.size()-1) out << ',';
        }
             out<< ',' << queryPoints[i].GetLabel() << ",query,"
                << i << ",0.0\n"; 
    }

    for(size_t i = 0; i < kNearest.size();i++){
        for(size_t j = 0; j < K_points;j++){
            const auto& Neighbors = kNearest[i][j];
            double dist = Neighbors.first;
            const auto& dp = Neighbors.second;
            const auto& feature = dp.GetFeatures();
            for(size_t j = 0; j < feature.size();j++){
                out << feature[j];
                if (j < feature.size()-1) out << ',';
            }
            out << ',' << dp.GetLabel() << ",neighbor,"
                << i << ','   
                << dist << "\n";
        }
    }
    for (size_t i = 0; i < Dataset.size();i++){
        const auto& feature = Dataset[i].GetFeatures();
        for(size_t j = 0; j < feature.size();j++){
            out << feature[j];
            if (j < feature.size()-1) out << ',';
        }
        out << ',' << Dataset[i].GetLabel() << ','
            << "dataset,-1,-1\n";
    }

    out.close();
    std::cout << "Successfully wrote data to KNN.csv";
}


double calculateDistance(const DataPoint&p1,const DataPoint&p2){
    double total =0;
    for (size_t i =0; i < p1.GetFeatures().size();i++){
        double difference = p2.GetFeatures()[i] - p1.GetFeatures()[i];
        double squared_diff = difference * difference;
        total += squared_diff;
    }
    return std::sqrt(total);
}

std::vector<std::pair<double, DataPoint>> findDistancePointPairs(const DataPoint& p1,const std::vector<DataPoint>&p2){
    std::vector<std::pair<double, DataPoint>> distancePointPairs;
    for(const auto& point:p2){
    double dist = calculateDistance(p1,point);
    distancePointPairs.emplace_back(dist,point);
    }
       return distancePointPairs;
}

void findKNN(std::vector<std::vector<std::pair<double, DataPoint>>> &distancePointPairs,int K_points){
    for(auto& distancePointPair:distancePointPairs){
    std::partial_sort(distancePointPair.begin(),
                      distancePointPair.begin()+K_points,
                      distancePointPair.end(),
                      [](const auto&a,const auto&b){
                          return a.first < b.first;         
                      }
                      );
    }
}

std::vector<DataPoint> generateRandomQueries(const std::vector<DataPoint>& dataset, int numQueries) {
    if (dataset.empty()) return {};
     
    size_t numFeatures = dataset[0].GetFeatures().size();
    std::vector<double> minVals(numFeatures, std::numeric_limits<double>::max());
    std::vector<double> maxVals(numFeatures, std::numeric_limits<double>::lowest());
    
    for (const auto& point : dataset) {
        const auto& features = point.GetFeatures();
        for (size_t i = 0; i < numFeatures; i++) {
            minVals[i] = std::min(minVals[i], features[i]);
            maxVals[i] = std::max(maxVals[i], features[i]);
        }
    }
    
    std::vector<DataPoint> queries;
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (int q = 0; q < numQueries; q++) {
        std::vector<double> features;
        for (size_t i = 0; i < numFeatures; i++) {
            std::uniform_real_distribution<double> dist(minVals[i], maxVals[i]);
            features.push_back(dist(gen));
        }
        queries.emplace_back(features, -1);
    }
    
    return queries;
}

int main (void){
    std::vector<DataPoint> datapoints = ReadCSV("dataset.csv");
    std::vector<DataPoint> queryPoints = generateRandomQueries(datapoints,100);  
    if (queryPoints[0].GetFeatures().size()!= datapoints[0].GetFeatures().size()) {
        std::cerr << "Query point features and Dataset features are not of the same size" << std::endl;
        return 0;
    }
    int K_points = 3;
    std::vector<std::vector<std::pair<double, DataPoint>>>distancePointPairs;
    for(auto& queryPoint:queryPoints){
        distancePointPairs.push_back(findDistancePointPairs(queryPoint,datapoints));
    }
    findKNN(distancePointPairs,K_points);
        
    
    for (int i = 0; i< distancePointPairs.size();i++) {

    std::unordered_map<int,int> voteCounts; 
        for (int j = 0; j < K_points;j++){
            int neighborLabel = distancePointPairs[i][j].second.GetLabel();
            voteCounts[neighborLabel]++;
        }
    

    auto index = std::max_element(voteCounts.begin(),voteCounts.end(),
                                  [](const auto& a,const auto&b){
                                      return a.second < b.second;
                                  });

    int predicted_label = index->first; 
    
    }
    WriteCSV (queryPoints,distancePointPairs, K_points, datapoints);  

}

    
