#include "HeaderTool.h"


void main()
{
	// Khung sườn khuôn mặt đứa bé để làm chuẩn kích thước ảnh output
	string ImgChild = "baby.jpg";
	std::vector<Vec6f> ChildTriList;

	cout << "\n\t-------------------- Processing --------------------";
	// Đoc ảnh CV_32F
	Mat ChildImage = imread(ImgChild);
	ChildImage.convertTo(ChildImage, CV_32F);



	// Lưu 68 điểm vào một vectơ, 68 điểm này là 68 điểm đánh dấu các bộ phận trên khuôn mặt
	// Kiểm tra có ảnh con đưa vô có duy nhất 1 khuôn mặt không. Nếu không thì báo lỗi
	std::vector<Point2i> ChildPoint;

	cout << "\n\t  Please wait...";
	if (getFeaturePoints(ChildPoint, ImgChild) == -1) {
		cout << "\n\tError: detector gets more than 2 faces.";
		cin.get();
		return;
	}
	cout << endl;

	// Phát sinh thêm tập dữ liệu  thêm vào danh sách các tọa độ điểm trên khuôn mặt
	getMorePoints(ChildPoint);

	///---------------------- Father ----------------------
	string ImgFather;
	std::vector<Vec6f> FaTriList;

	/// Nhập ảnh bố
	cout << "\n\t---------------------- Father ----------------------";
	cout << "\n\t  Input: ";
	getline(cin, ImgFather);


	/// Đoc ảnh CV_32F
	Mat FaImage = imread(ImgFather);
	while (FaImage.empty())
	{
		cout << "Sai duong dan! Moi nhap lai!";
		cout << "\n\t  Input: ";
		getline(cin, ImgFather);
		FaImage = imread(ImgFather);
	}
	FaImage.convertTo(FaImage, CV_32F);

	// Lưu 68 điểm vào một vectơ, 68 điểm này là 68 điểm đánh dấu các bộ phận trên khuôn mặt
	// Kiểm tra có ảnh con đưa vô có duy nhất 1 khuôn mặt không. Nếu không thì báo lỗi
	std::vector<Point2i> FatherPoint;

	cout << "\n\t  Please wait...";
	if (getFeaturePoints(FatherPoint, ImgFather) == -1) {
		cout << "\n\tError: detector gets more than 2 faces.";
		cin.get();
		return;
	}

	///---------------------- Mother ----------------------
	string ImgMother;
	std::vector<Vec6f> MoTriList;

	cout << "\n\t---------------------- Mother ----------------------";
	cout << "\n\t  Input: ";
	getline(cin, ImgMother);

	//  Nhập ảnh mẹ
	Mat MoImage = imread(ImgMother);
	while (MoImage.empty())
	{
		cout << "Sai duong dan! Moi nhap lai!";
		cout << "\n\t  Input: ";
		getline(cin, ImgMother);
		MoImage = imread(ImgMother);
	}
	MoImage.convertTo(MoImage, CV_32F);

	// Lưu 68 điểm vào một vectơ, 68 điểm này là 68 điểm đánh dấu các bộ phận trên khuôn mặt
	// Kiểm tra có ảnh con đưa vô có duy nhất 1 khuôn mặt không. Nếu không thì báo lỗi
	std::vector<Point2i> MotherPoint;

	cout << "\n\t  Please wait...";
	if (getFeaturePoints(MotherPoint, ImgMother) == -1) {
		cout << "\n\tError: detector gets more than 2 faces.";
		cin.get();
		return;
	}


	// Phát sinh thêm tập dữ liệu  thêm vào danh sách các tọa độ điểm trên khuôn mặt bố
	getMorePoints(FatherPoint);

	// Phát sinh thêm tập dữ liệu  thêm vào danh sách các tọa độ điểm trên khuôn mặt mẹ
	getMorePoints(MotherPoint);

	/// Phép đổi hình ảnh
	Mat ImgMorph = Mat::zeros(FaImage.size(), CV_32FC3);
	std::vector<Point2f> Points;

	cout << "\n\t----------------------  Baby  ----------------------";

	cout << "\n\tInput alpha: \n";
	double Alpha = 0.5;

	do {
		cout << "\n\tInput: ";
		cin >> Alpha;
	} while (Alpha <= 0 || Alpha >= 1);


	//Tính vị trí của các điểm đặc trưng trên ảnh Morphed Image  tương ứng với với vị trí pixel tương ứng của ảnh bố và mẹ
	for (int i = 0; i < FatherPoint.size(); i++) {
		float X, Y;
		X = Alpha * FatherPoint[i].x + (1 - Alpha) * MotherPoint[i].x;
		Y = Alpha * FatherPoint[i].y + (1 - Alpha) * MotherPoint[i].y;

		// Push point
		Points.push_back(Point2f(X, Y));
	}

	int Vertex1, Vertex2, Vertex3;

	///---------------------- Morphing ----------------------
	Mat ImgBaby = Mat::zeros(FaImage.size(), CV_32FC3);

	// Đọc 3 điểm trong file Triangulation.txt để biến đổi vector khuôn mặt của bố, mẹ và ảnh con ban đầu, file  Triangulation.txt la file lưu vị trí các điểm tương ứng để thõa  Delaunay Triangulation
	ifstream FileTriMini("Triangulation.txt");

	while (FileTriMini >> Vertex1 >> Vertex2 >> Vertex3) {
		// Danh sách các điểm lưu các tam giác được phát sinh từ các điểm đặc trưng
		std::vector<Point2f> TriangleFather, TriangleMother, TriangleMorph;

		// Tạo tam giác từ các điểm đặc trưng thõa  Delaunay Triangulation cho ảnh bố
		TriangleFather.push_back(FatherPoint[Vertex1]);
		TriangleFather.push_back(FatherPoint[Vertex2]);
		TriangleFather.push_back(FatherPoint[Vertex3]);

		// Tạo tam giác từ các điểm đặc trưng thõa  Delaunay Triangulation cho ảnh mẹ
		TriangleMother.push_back(MotherPoint[Vertex1]);
		TriangleMother.push_back(MotherPoint[Vertex2]);
		TriangleMother.push_back(MotherPoint[Vertex3]);

		// Tạo tam giác từ các điểm đặc trưng thõa  Delaunay Triangulation cho ảnh Morphed Image
		TriangleMorph.push_back(ChildPoint[Vertex1]);
		TriangleMorph.push_back(ChildPoint[Vertex2]);
		TriangleMorph.push_back(ChildPoint[Vertex3]);

		// Phát sinh ảnh mang đặc trưng của ảnh bố và ảnh mẹ
		morphBabyFromParents(FaImage, MoImage, ImgBaby, TriangleFather, TriangleMother, TriangleMorph, Alpha);
	}



	FileTriMini.close();

	/// Hiển thị ảnh bố
	namedWindow("Father - BabyFaceGenerator", WINDOW_NORMAL);
	imshow("Father - BabyFaceGenerator", FaImage / 255.0);
	resizeWindow("Father - BabyFaceGenerator", FaImage.size().width*0.5, FaImage.size().height*0.5);

	/// Hiển thị ảnh con
	namedWindow("Mother - BabyFaceGenerator", WINDOW_NORMAL);
	imshow("Mother - BabyFaceGenerator", MoImage / 255.0);
	resizeWindow("Mother - BabyFaceGenerator", FaImage.size().width*0.5, FaImage.size().height*0.5);
	
	// Từ ảnh mang đặc trưng của bố mẹ, biến đổi về kich thước của ảnh con
	Rect RectBabyFace = boundingRect(ChildPoint);
	Mat ImageBabyNew = ImgBaby(RectBabyFace);

	// Hiển thị ảnh con 
	namedWindow("Baby - BabyFaceGenerator", WINDOW_NORMAL);
	imshow("Baby - BabyFaceGenerator", ImageBabyNew / 255.0);
	resizeWindow("Baby - BabyFaceGenerator", ImageBabyNew.size().width*0.75, ImageBabyNew.size().height*0.75);
	
	// Lưu lại hình ảnh con sau khi phát sinh
	for (int i = 0; i < 4; i++)
	{
		ImgFather.pop_back();
		ImgMother.pop_back();
	}
	imwrite("s-" + ImgFather + "-" + ImgMother + ".jpg", ImageBabyNew);

	waitKey(0);
}