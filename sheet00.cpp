#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


#include <fstream>
#include <cstdio>
#include <climits>

#include <iterator>
#include <vector>
#include <stdint.h>
#include <iomanip>
#include <math.h>


using namespace std;
using namespace cv;


template <typename T>
long atoi(const std::basic_string<T> &str){
	std::basic_stringstream<T> stream(str);
	long res;
	return !(stream >>res)?0:res;
}

double  interpolate(double value, double leftMin, double leftMax, double rightMin, double rightMax){
    double leftSpan = leftMax - leftMin;
    double rightSpan = rightMax - rightMin;

    double valueScaled = (value - leftMin) / (leftSpan);

    return rightMin + (valueScaled * rightSpan);
  }

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

Mat read_ir (std::string filename, int clip){



  int rows_ir=424;
  int cols_ir=512;

    Mat image = Mat::zeros(rows_ir,cols_ir, CV_32S);

    //Read txt
    vector <vector <string> > data;
    ifstream infile(filename );
    while (infile) {
      string s;
      if (!getline( infile, s )) break;
      istringstream ss( s );
      vector <string> record;

      while (ss) {
          string s;
          if (!getline( ss, s, ',' )) break;
          record.push_back( s );
          // std::cout <<  "val is" << atoi(s) << std::endl;
      }

      data.push_back( record );
    }
    if (!infile.eof()){
      cerr << "Fooey!\n";
    }


  for (size_t i = 0; i < rows_ir; i++) {
     for (size_t j = 0; j < cols_ir; j++) {
         int32_t val= atoi (data[i][j]);


         if (clip!=0){
           if (val >clip){
               val=clip;
           }
         }

         image.at<int32_t>(i,j)=  val;
      //    std::cout << "added" << image.at<int32_t>(i,j) << std::endl;
     }
   }



   //map the values to 0-255
   double min;
   double max;
   cv::minMaxIdx(image, &min, &max);
   cv::Mat adjMap;
   // expand your range to 0..255. Similar to histEq();
   image.convertTo(adjMap,CV_8UC1, 255 / (max-min), -min);
  //  applyColorMap(adjMap, adjMap, cv::COLORMAP_BONE);
  //  applyColorMap(adjMap, adjMap, cv::COLORMAP_AUTUMN);

   return adjMap;


}



int main(int argc, char* argv[])
{
    int rows_ir=424;
    int cols_ir=512;

    std::vector <Mat> irpic;
    irpic.resize(9);

    Mat image = Mat::zeros(rows_ir,cols_ir, CV_32S);



      //read binary 3
    //   int32_t i;
    //   vector<int32_t> ivector;
    //   char *filename = "/media/alex/Nuevo_vol/Master/2nd Semester/Computacional_Photography/Project/Code/libfreenect2-/build-qt/bin/Frames/20160616-120945411.ir";
    //   ifstream infile(filename,ios::binary);
    //   if (!infile) {
    //       cerr << "Can't open file " << filename << " for reading" << endl;
    //       exit(EXIT_FAILURE);
    //   }
    //   cout << "Opened binary file " << filename << " for reading." << endl;
    //   while (infile.read(reinterpret_cast<char *>(&i), sizeof(i))) {
    //        ivector.push_back(i);
    //    }
    //   for (unsigned count = 0; count < ivector.size(); count++) {
    //         cout << "ivector[" << count << "] = " << setw(11) << ivector[count] << endl;
    //     }
    //     infile.close();
      //
      //
    //     for (size_t i = 0; i < rows_ir; i++) {
    //            for (size_t j = 0; j < cols_ir; j++) {
    //                image.at<double>(i,j)= std::abs( ivector[i*j]);
    //            }
    //        }
      //


      //Read the irpic images
      for (size_t i = 0; i < 9; i++) {
          std::string prefix = "/media/alex/Nuevo_vol/Master/2nd Semester/Computacional_Photography/Project/Code/libfreenect2-/build-qt/bin/Frames/ir_";
          std::string index = std::to_string(i);
          std::string sufix= ".out";

          std::string filename= prefix + index + sufix;

          std::cout << "reading " << filename << std::endl;
          irpic[i]=read_ir(filename,500);
      }


      //show them
      for (size_t i = 0; i < 9; i++) {
        // cv::imshow("Out", irpic[i]);
        // waitKey(0); //wait infinite time for a keypress
      }

      std::cout << "finished reading raw images" << std::endl;

      std::cout << "images are type" <<  type2str( irpic[0].type() ) << std::endl;;

      std::vector<Mat> phases;
      for (size_t i = 0; i < 3; i++) {
          Mat image = Mat::zeros(rows_ir,cols_ir, CV_64F);
          phases.push_back(image);
      }

      std::cout << "calculating phasses" << std::endl;

      Mat phase_255 = Mat::zeros(rows_ir,cols_ir, CV_8UC1);

      //calculate the first phases
      double max=-100.0;
      double min=10000.0;
      for (size_t i = 0; i < rows_ir; i++) {
         for (size_t j = 0; j < cols_ir; j++) {
             double voltage_1= irpic[0].at<unsigned char>(i,j);
             double voltage_2= irpic[1].at<unsigned char>(i,j);
             double voltage_3= irpic[2].at<unsigned char>(i,j);

             double term_sin= - voltage_1 * sin(0) - voltage_2 * sin (2*M_PI/3) - voltage_3*sin (4*M_PI/3);
             double term_cos=   voltage_1 * cos(0) + voltage_2 * cos (2*M_PI/3) + voltage_3*cos (4*M_PI/3);

             double phase =atan2 (term_sin, term_cos);

            //  std::cout << "phase is" << phase << std::endl;

             if (phase > max)
                max= phase;
             if (phase < min)
                min=phase;


             phases[0].at<double>(i,j)=interpolate(phase, -M_PI, M_PI, 0.0, 1.0);

         }
       }

      std::cout << "max is" << max << std::endl;
      std::cout << "min is" << min << std::endl;

      cv::imshow("Out", phases[0]);
      waitKey(0); //wait infinite time for a keypress

      // cv::imshow("Out", phase_255);
      // waitKey(0); //wait infinite time for a keypress


    return 0;
}
