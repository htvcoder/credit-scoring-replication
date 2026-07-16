# Học sâu cho chấm điểm tín dụng: Nên hay không?

**Björn Rafn Gunnarsson, Seppe vanden Broucke, Bart Baesens, María Óskarsdóttir, Wilfried Lemahieu**

Trung tâm Nghiên cứu Kỹ thuật Hệ thống Thông tin (LIRIS), KU Leuven; Khoa Tin học Kinh doanh và Quản trị Vận hành, Ghent University; Khoa Phân tích Quyết định và Rủi ro, University of Southampton; Khoa Khoa học Máy tính, Reykjavík University.

> Bản tiền ấn phẩm gửi tới *European Journal of Operational Research*, ngày 4 tháng 3 năm 2021.

## Tóm tắt

Việc phát triển các mô hình phân tích chấm điểm tín dụng chính xác đã trở thành một trọng tâm lớn đối với các định chế tài chính. Vì mục đích này, nhiều thuật toán phân loại đã được đề xuất cho chấm điểm tín dụng. Tuy nhiên, việc áp dụng các thuật toán học sâu cho phân loại hầu như đã bị bỏ qua trong văn liệu về chấm điểm tín dụng. Động lực chính của nghiên cứu này là xem xét mức độ phù hợp của các thuật toán học sâu đối với chấm điểm tín dụng. Để đạt mục tiêu đó, hai kiến trúc học sâu được xây dựng, cụ thể là mạng perceptron đa tầng và mạng niềm tin sâu, rồi hiệu năng của chúng được so sánh với hai phương pháp truyền thống và hai phương pháp tổ hợp cho chấm điểm tín dụng. Các mô hình được đánh giá bằng nhiều bộ dữ liệu chấm điểm tín dụng và các thước đo hiệu năng khác nhau. Hơn nữa, các thủ tục kiểm định thống kê Bayes được giới thiệu trong bối cảnh chấm điểm tín dụng và được so sánh với các thủ tục kiểm định phi tham số theo trường phái tần suất, vốn theo truyền thống được xem là thực hành tốt nhất trong chấm điểm tín dụng. Hai kết luận chính được rút ra. Thứ nhất, phương pháp tổ hợp XGBoost là phương pháp có hiệu năng tốt nhất trong các phương pháp được xem xét. Thứ hai, các mạng nơ-ron sâu không vượt trội hơn các đối ứng nông hơn và tốn kém hơn đáng kể về mặt tính toán để xây dựng. Do đó, trong phép so sánh này, các thuật toán học sâu dường như không phải là những mô hình phù hợp cho chấm điểm tín dụng, và nên ưu tiên XGBoost khi hiệu năng phân loại là mục tiêu chính.

**Từ khóa:** hệ thống hỗ trợ quyết định; phân tích rủi ro; chấm điểm tín dụng; học sâu; kiểm định thống kê Bayes

## 1. Giới thiệu

Từ giữa thế kỷ 20, cả giới nghiên cứu lẫn những người thực hành đã nhấn mạnh mạnh mẽ việc phát triển các mô hình thực nghiệm nhằm hỗ trợ các bên cho vay bán lẻ trong quyết định cấp tín dụng cho người tiêu dùng. Ngày nay, các ngân hàng thương mại nắm giữ hàng tỷ đô la trong các khoản vay tiêu dùng và lĩnh vực tín dụng tiêu dùng đã trở thành một ngành công nghiệp lớn có tầm quan trọng kinh tế đáng kể. Quy mô lớn của các khoản vay này cho thấy ngay cả những cải thiện nhỏ về độ chính xác của thực hành chấm điểm tín dụng cũng có thể dẫn đến lợi ích tài chính đáng kể. Vì vậy, phát triển các mô hình chấm điểm tín dụng chính xác đã trở thành một trọng tâm lớn của các định chế tài chính nhằm tối ưu hóa lợi nhuận và quản trị hiệu quả mức độ phơi nhiễm rủi ro (Thomas et al., 2002; Lessmann et al., 2015; Board of Governors of the Federal Reserve System, 2019; Baesens et al., 2003; Jiang et al., 2019; Luo et al., 2017). Trong lịch sử, các nhà quản lý thường đánh giá tín dụng của người tiêu dùng dựa trên kinh nghiệm trực giác của họ. Tuy nhiên, với sự hỗ trợ của các mô hình thực nghiệm, các nhà quản lý có thể đánh giá người xin cấp tín dụng theo cách nhanh hơn, nhất quán hơn và chính xác hơn. Hệ quả là chấm điểm tín dụng thực nghiệm được chú ý nhiều hơn, dẫn tới sự phát triển của một số kỹ thuật thường được gọi là “mô hình chấm điểm tín dụng”. Mục tiêu của các mô hình này là phân loại người xin cấp tín dụng vào nhóm người xin cấp tín dụng tốt, tức những người có khả năng hoàn trả khoản vay, hoặc nhóm người xin cấp tín dụng xấu, tức những người có khả năng vỡ nợ đối với khoản vay của họ. Do đó, các bài toán chấm điểm tín dụng có thể được đặt trong phạm vi của các bài toán phân loại được thảo luận rộng rãi hơn (Baesens et al., 2016; Verbraken et al., 2014; Saberi et al., 2013; Akkoç, 2012).

Kể từ khi hình thành, nhiều kỹ thuật phân loại khác nhau đã được đề xuất và sử dụng cho chấm điểm tín dụng, bao gồm các mô hình thống kê truyền thống (ví dụ: hồi quy logistic), các mô hình bắt nguồn từ học máy (ví dụ: cây quyết định) và mạng nơ-ron (Baesens et al., 2003). Hiệu năng của các thuật toán phân loại khác nhau cho chấm điểm tín dụng đã được nghiên cứu sâu rộng trong những thập kỷ qua, và một số nghiên cứu đã xem xét hiệu năng của các bộ phân loại riêng lẻ thay thế cho chấm điểm tín dụng (ví dụ: Baesens et al. (2003); Xiao et al. (2006); Huang et al. (2006); Yeh and Lien (2009)). Gần đây hơn, nghiên cứu về các thuật toán phân loại cho chấm điểm tín dụng đã tính đến sự phát triển của các phương pháp tổ hợp nhằm ước lượng nhiều mô hình phân tích kết hợp thay vì chỉ xây dựng một mô hình (Baesens, 2014). Khá nhiều nghiên cứu đã xem xét hiệu năng của các thuật toán tổ hợp khác nhau cho chấm điểm tín dụng (ví dụ: Zhou et al. (2010); Yu et al. (2011); Marqúes et al. (2012); Lessmann et al. (2015); Chen et al. (2020)). Tuy nhiên, trong công trình này, chúng tôi lập luận rằng nghiên cứu về các thuật toán phân loại cho chấm điểm tín dụng phần lớn đã bỏ qua một phát triển quan trọng khác trong học máy. Đó là sự phát triển của các cách tiếp cận được gọi là “học sâu”, vốn đã được nghiên cứu và ứng dụng rộng rãi trong nhiều lĩnh vực với thành công lớn (để có tổng quan chi tiết về các cách tiếp cận và ứng dụng học sâu, xem ví dụ Schmidhuber (2015); LeCun et al. (2015); Goodfellow et al. (2016)). Vì vậy, mục đích chính của công trình này là bổ sung vào khối nghiên cứu hiện có trong cộng đồng chấm điểm tín dụng bằng cách xem xét các thuật toán học sâu mới cho chấm điểm tín dụng. Khi làm như vậy, các đóng góp sau đây sẽ được thực hiện cho văn liệu chấm điểm tín dụng. Thứ nhất, các kỹ thuật học sâu tiên tiến được so sánh với cả các phương pháp truyền thống cho chấm điểm tín dụng và hai phương pháp tổ hợp đã được chứng minh là hoạt động tốt trong chấm điểm tín dụng. Thứ hai, phép so sánh này sẽ được thực hiện trên một số lượng đáng kể các bộ dữ liệu chấm điểm tín dụng đời thực. Thứ ba, các mô hình sẽ được đánh giá và so sánh theo một số thước đo hiệu năng, bao gồm một thước đo hiệu năng tổng quát định hướng lợi nhuận. Cuối cùng, kiểm định giả thuyết Bayes sẽ được giới thiệu trong bối cảnh chấm điểm tín dụng và được so sánh với một thủ tục kiểm định thống kê phi tham số tần suất nâng cao. Thủ tục sau theo truyền thống được xem là thực hành tốt nhất để so sánh nhiều bộ phân loại trên nhiều bộ dữ liệu trong cộng đồng chấm điểm tín dụng. Tuy nhiên, các thủ tục kiểm định thống kê tần suất ngày càng bị xem xét kỹ lưỡng hơn và đã mất dần vị thế trong nhiều lĩnh vực khoa học (xem ví dụ Wasserstein et al. (2016); Benavoli et al. (2017)). Do đó, phép so sánh này sẽ làm sáng tỏ sự khác biệt giữa hai trường phái tư duy khi nói đến các thủ tục kiểm định thống kê, đồng thời nhấn mạnh nhiều lợi ích của các thủ tục thống kê Bayes bên cạnh việc củng cố các phát hiện thực nghiệm.

Phần còn lại của bài báo được cấu trúc như sau. Mục 2 cung cấp tổng quan về sự phát triển của chấm điểm tín dụng định lượng và văn liệu liên quan. Mục 3 phác thảo các phương pháp truyền thống đã được thiết lập và hai phương pháp tổ hợp cho chấm điểm tín dụng. Ngoài ra, phần này cũng mô tả các kiến trúc học sâu được xem xét trong công trình này. Các khía cạnh quan trọng liên quan đến thiết kế thực nghiệm được sử dụng trong dự án này, chẳng hạn thông tin về các phương pháp tiền xử lý và các chỉ báo hiệu năng, được thảo luận trong Mục 4. Phần này cũng cung cấp thảo luận về cả các thủ tục kiểm định giả thuyết theo trường phái tần suất và Bayes. Mục 5 sau đó trình bày các kết quả thực nghiệm và thảo luận về các phát hiện. Cuối cùng, mục cuối cùng đưa ra các nhận xét kết luận và cơ hội cho nghiên cứu tiếp theo.

## 2. Chấm điểm tín dụng

Việc sử dụng chấm điểm tín dụng thống kê đầu tiên có thể được ghi nhận cho Durand (1941) vào giữa thế kỷ 20. Trong giai đoạn đó, các phương pháp được sử dụng là phân biệt thống kê và phân loại. Ngày nay, các phương pháp này vẫn là những phương pháp phổ biến nhất để xây dựng thẻ điểm tín dụng. Hồi quy logistic là phương pháp được sử dụng rộng rãi nhất trong số đó, mặc dù cây quyết định hoặc cây phân loại cũng đã được ưa chuộng trong 30 năm qua (Thomas et al., 2002; Lessmann et al., 2015). Nhiều kỹ thuật phân loại khác đã được áp dụng cho chấm điểm tín dụng kể từ lần sử dụng đầu tiên cách đây hơn 70 năm. Năm 2015, Lessmann et al. (2015) thực hiện một nghiên cứu toàn diện có tính đến một số tiến bộ trong học máy, ví dụ sự phát triển của các mô hình tổ hợp. Để đạt mục tiêu đó, 41 bộ phân loại được so sánh theo sáu thước đo hiệu năng. Nghiên cứu kết luận rằng một số bộ phân loại dự đoán rủi ro tín dụng tốt hơn đáng kể so với chuẩn ngành, cụ thể là hồi quy logistic. Đặc biệt, nghiên cứu khuyến nghị rằng rừng ngẫu nhiên nên được xem là kỹ thuật phân loại chuẩn để so sánh với các thuật toán phân loại mới, thay vì hồi quy logistic vốn theo truyền thống giữ vị trí đó. Ngoài ra, nghiên cứu cũng phát hiện rằng cả rừng ngẫu nhiên và mạng nơ-ron nhân tạo đều đạt được mức giảm chi phí lớn khi chi phí của lỗi phân loại được ước lượng, cho thấy lợi ích của hai bộ phân loại này. Cuối cùng, nghiên cứu chuẩn khuyến nghị rằng các nghiên cứu tương lai nên sử dụng ít nhất ba thước đo hiệu năng để đánh giá hiệu năng dự đoán của các thuật toán chấm điểm tín dụng, đó là diện tích dưới đường cong đặc trưng hoạt động của bộ thu nhận (ROC) (AUC), Gini từng phần và Brier Score, vì tất cả các thước đo này nắm bắt những khía cạnh khác nhau của hiệu năng bộ phân loại (Lessmann et al., 2015). Kể từ khi Lessmann et al. (2015) công bố nghiên cứu chuẩn, các phương pháp tổ hợp khác đã được đề xuất trong văn liệu. Đáng chú ý nhất, Chen and Guestrin (2016) đề xuất XGBoost, phương pháp sử dụng tăng cường gradient cực đại để học một tổ hợp cây quyết định. Phương pháp này đã đạt kết quả hứa hẹn trên một số tác vụ phân loại (Xia et al., 2017), bao gồm chấm điểm tín dụng, nơi ví dụ Wang et al. (2018b) phát hiện phương pháp này vượt trội hơn rừng ngẫu nhiên trong dự đoán rủi ro tín dụng.

Như đã thảo luận ở trên, nghiên cứu chuẩn do Lessmann et al. (2015) công bố phát hiện rằng một mạng nơ-ron nhân tạo hoạt động tốt khi chi phí của lỗi phân loại được ước lượng. Kỷ nguyên hiện đại của nghiên cứu về các mạng này bắt đầu với công trình tiên phong của McCulloch and Pitts (1943), trong đó chỉ ra rằng về mặt lý thuyết các mạng như vậy có thể khớp bất kỳ hàm tính toán được nào. Với kết quả quan trọng này, nhìn chung người ta đồng thuận rằng các ngành mạng nơ-ron và trí tuệ nhân tạo đã ra đời. Nhờ các tiến bộ trong những lĩnh vực này và sự cải thiện hiệu năng tính toán, việc phát triển các mạng nơ-ron lớn với nhiều tầng nơ-ron, tức học sâu, đã trở nên khả thi. Các mạng nơ-ron sâu đã là đối tượng của nghiên cứu rộng rãi trong học máy và đã đạt thành công lớn trong một số lĩnh vực như thị giác máy tính, nhận dạng tiếng nói và phân loại (Haykin, 1994; LeCun et al., 2015; Luo et al., 2017; Spanoudes and Nguyen, 2017; Deng, 2014). Gần đây, việc ứng dụng các mô hình này cho

**Bảng 1. Tổng quan văn liệu liên quan phân tích hiệu năng tương đối của các cách tiếp cận học sâu so với các phương pháp phân loại khác cho chấm điểm tín dụng.**

| Tác giả (Năm) | Bộ dữ liệu | Kiến trúc DL | Bộ phân loại khác | Chỉ báo hiệu năng | Kiểm định giả thuyết thống kê |
|---|---:|---|---|---|:---:|
| Van-Sang and Ha-Nam (2016) | 2 | MLP | IND, ENS | TH |  |
| Luo et al. (2017) | 1 | DBN | IND | TH, AUC |  |
| Addo et al. (2018) | 1 | MLP | IND, ENS | TH, AUC, RMSE |  |
| Zhu et al. (2018) | 1 | CNN | IND, ENS | TH, AUC, K-S statistic |  |
| Hamori et al. (2018) | 1 | MLP | ENS | TH, AUC |  |
| Sun and Vasarhelyi (2018) | 1 | MLP | IND | TH, AUC | Có |
| Wang et al. (2018a) | 1 | LSTM | ENS | AUC, K-S statistic |  |
| Papouskova and Hajek (2019) | 2 | DBN | IND, ENS | TH, AUC | Có |
| Munkhdalai et al. (2019) | 3 | MLP | IND | TH, AUC, H-measure |  |
| Mancisidor et al. (2019) | 2 | DGM | IND | TH, AUC, H-measure, Gini |  |

*IND: bộ phân loại riêng lẻ (ví dụ: hồi quy logistic); ENS: phương pháp tổ hợp (ví dụ: rừng ngẫu nhiên); TH: chỉ số theo ngưỡng (ví dụ: độ chính xác).*

phân tích kinh doanh và nghiên cứu vận hành ngày càng được khảo sát. Ví dụ, Kraus et al. (2020) phát hiện học sâu là một phương pháp khả thi và hiệu quả trong các lĩnh vực đó, có thể nhất quán vượt trội hơn các đối ứng truyền thống về cả hiệu năng dự đoán lẫn hiệu năng vận hành. Tuy nhiên, số lượng nghiên cứu đã công bố về việc áp dụng học sâu cho chấm điểm tín dụng còn hạn chế. Tổng quan các nghiên cứu trước đây trong đó hiệu năng của các cách tiếp cận học sâu được so sánh với hiệu năng của các phương pháp phân loại khác cho chấm điểm tín dụng được trình bày trong Bảng 1. Như có thể thấy từ bảng, các kiến trúc học sâu khác nhau đã được xem xét cho chấm điểm tín dụng. Trong Wang et al. (2018a), các tác giả sử dụng hành vi tương tác trực tuyến (ví dụ: sự kiện duyệt và nhấp của người dùng) thu được từ nhật ký sự kiện để xây dựng một mạng bộ nhớ dài ngắn hạn (LSTM) cho chấm điểm tín dụng. Cũng trong năm đó, Zhu et al. (2018) đề xuất một mô hình lai, trong đó hồ sơ tín dụng của khách hàng được chuyển thành ma trận điểm ảnh và sau đó các ma trận thu được được dùng để xây dựng mạng nơ-ron tích chập (CNN) nhằm dự đoán vỡ nợ. Gần đây hơn, Mancisidor et al. (2019) xây dựng một mô hình sinh sâu (DGM) với mục tiêu cải thiện độ chính xác phân loại của các mô hình chấm điểm tín dụng bằng cách bổ sung các hồ sơ bị từ chối. Tuy nhiên, chủ yếu có hai kiến trúc học sâu trước đây đã được xây dựng cho chấm điểm hồ sơ vay sử dụng thiết lập tiền xử lý chuẩn, cụ thể là mạng perceptron đa tầng (MLP) và mạng niềm tin sâu (DBN). Một hạn chế tiềm tàng của các bài báo liệt kê trong bảng là việc sử dụng số lượng bộ dữ liệu nhỏ khi đánh giá hiệu năng của các kiến trúc học sâu, ví dụ phần lớn các bài báo chỉ sử dụng một bộ dữ liệu để xây dựng và so sánh các bộ phân loại được xem xét. Hơn nữa, hầu hết các bài báo không tính đến các cân nhắc hiệu năng quan trọng như tính đúng đắn của các dự đoán thực tế của bộ phân loại và lợi nhuận mà một công ty có thể đạt được khi áp dụng một bộ phân loại cụ thể, được tính bằng cách xét cả lợi ích từ việc phân loại đúng một khách hàng sẽ vỡ nợ và chi phí của việc phân loại một khách hàng không vỡ nợ thành người vỡ nợ. Cuối cùng, chỉ hai trong số các bài báo sử dụng kiểm định giả thuyết thống kê khi so sánh hiệu năng của các bộ phân loại được xem xét. Cũng cần lưu ý ở đây rằng tính áp dụng được của học sâu cho chấm điểm tín dụng vẫn là một câu hỏi mở. Trong số bảy bài báo mà các thuật toán học sâu được xem xét cho chấm điểm hồ sơ vay bằng thiết lập tiền xử lý chuẩn, bốn bài kết luận rằng cách tiếp cận học sâu được xem xét là phương pháp có hiệu năng tốt nhất, trong khi ba bài kết luận rằng nên ưu tiên một cách tiếp cận tổ hợp cho chấm điểm tín dụng. Do đó, trong công trình này, chúng tôi hướng tới mở rộng hiện trạng nghiên cứu bằng cách (i) so sánh hiệu năng của các kiến trúc học sâu tiên tiến với hiệu năng của các bộ phân loại truyền thống và tổ hợp trên một số lượng bộ dữ liệu chấm điểm tín dụng chưa từng có, (ii) đánh giá hiệu năng của các bộ phân loại theo một số chỉ báo hiệu năng, bao gồm một thước đo hiệu năng định hướng lợi nhuận, và (iii) sử dụng các thủ tục kiểm định thống kê mới và phù hợp để củng cố các phát hiện thực nghiệm.

## 3. Các phương pháp cho chấm điểm tín dụng

Như đã thảo luận trong mục trước, các phương pháp chấm điểm tín dụng được phát triển với mục tiêu phân biệt chính xác người xin cấp tín dụng tốt với người xin cấp tín dụng xấu. Để mô tả quá trình phát triển một thẻ điểm tín dụng, gọi D = {(x1, y1),..., (xn, yn)} là một tập gồm n ví dụ huấn luyện với xi = (xi 1,..., xi m) ∈Rm mô tả vectơ đầu vào của ví dụ thứ i, biểu diễn m đặc điểm hay “thuộc tính” của hồ sơ (chẳng hạn thông tin về người nộp đơn, loại khoản vay, v.v.)—hoặc đơn giản là x như một ký hiệu viết tắt để chỉ một thể hiện, khi đó xj biểu thị một đầu vào đơn lẻ. Gọi yi ∈{−1, +1} là một biến nhị phân phân biệt khoản vay tốt (-1) với khoản vay xấu (+1) (hoặc đơn giản là y như ký hiệu viết tắt). Thẻ điểm tín dụng là một mô hình thu được từ việc áp dụng thuật toán phân loại vào một bộ dữ liệu các khoản vay trong quá khứ. Mô hình ước lượng xác suất ˆy = $p(+1\mid x)$ rằng vỡ nợ sẽ được quan sát đối với một khoản vay nhất định. Sau đó, để quyết định liệu người xin vay có được xem là đủ uy tín tín dụng hay không, xác suất vỡ nợ ước lượng được so sánh với một ngưỡng t và khoản vay được phê duyệt nếu $p(+1\mid x)$ ≤t, nếu không thì bị từ chối (Lessmann et al., 2015; Thomas et al., 2002; Maldonado et al., 2017). Dựa trên tổng quan về hiện trạng nghiên cứu ở trên, chúng tôi kết luận rằng cần có thêm nghiên cứu về mức độ phù hợp của học sâu đối với chấm điểm tín dụng. Dựa trên rà soát các bài báo đã công bố về học sâu cho chấm điểm tín dụng, có thể thấy hai kiến trúc học sâu, mạng nơ-ron perceptron đa tầng và mạng niềm tin sâu, trước đây đã được sử dụng trong bối cảnh ứng dụng này. Do đó, chúng được so sánh với hai phương pháp tổ hợp đã được chứng minh là hoạt động tốt cho chấm điểm tín dụng, cụ thể là rừng ngẫu nhiên và XGBoost, cùng hai phương pháp truyền thống cho chấm điểm tín dụng, cụ thể là hồi quy logistic và cây quyết định. Các phương pháp này được thảo luận chi tiết hơn dưới đây.

### 3.1. Các phương pháp truyền thống cho chấm điểm tín dụng

Trong những thập kỷ qua, hồi quy logistic đã trở thành phương pháp phân tích tiêu chuẩn trong nhiều lĩnh vực mà biến kết quả quan tâm là một biến nhị phân rời rạc (Hosmer Jr et al., 2013). Với một tập huấn luyện, hồi quy logistic ước lượng xác suất vỡ nợ, $p(+1\mid x)$, đối với một khoản vay x như sau:

$$
p(+1\mid x)=\frac{1}{1+\exp\left[-\left(w_0+w^T x\right)\right]}
\tag{1}
$$

trong đó w là vectơ tham số và đại lượng vô hướng w0 là hệ số chặn (Baesens et al., 2003).

Các thuật toán cây quyết định là các thuật toán phân loại áp dụng phân hoạch đệ quy trên một bộ dữ liệu nhất định để tạo ra một cấu trúc dạng cây biểu diễn các mẫu trong dữ liệu nền bằng cách sắp xếp chúng dựa trên giá trị của các biến có trong dữ liệu. Cây quyết định nhằm phân hoạch bộ dữ liệu thành các nhóm đồng nhất nhất có thể theo biến cần dự đoán. Nhiều thuật toán cây quyết định đã được đề xuất trong văn liệu. Thuật toán C4.5 là một trong những thuật toán phổ biến nhất và sử dụng entropy để tính mức độ đồng nhất trong một mẫu nhằm quyết định việc phân hoạch. Sau đó, thuật toán ưu tiên một cách tham lam các phép tách có độ tăng entropy chuẩn hóa lớn nhất. Cây sau đó được xây dựng bằng cách lặp lại đệ quy quy trình này trên các tập con được tạo ra. Phương pháp này thường tạo ra một cấu trúc cây phức tạp với nhiều nút bên trong, có thể dẫn đến một nghiệm khớp quá mức dữ liệu, tức mô hình bắt đầu mô hình hóa nhiễu trong dữ liệu. Để khắc phục điều này, thuật toán cắt tỉa cây kết quả sau khi cây đã được phát triển đầy đủ bằng cách loại bỏ các nút phát sinh từ nhiễu trong mẫu huấn luyện (Baesens et al., 2003; Baesens, 2014; Sharma et al., 2013; Hssina et al., 2014).

### 3.2. Các phương pháp tổ hợp cho chấm điểm tín dụng

Các phương pháp tổ hợp ước lượng nhiều mô hình thay vì chỉ sử dụng một mô hình. Rừng ngẫu nhiên là một phương pháp như vậy do Breiman (2001) đề xuất và đã cho thấy hiệu năng cao trong nhiều lĩnh vực. Ở đây, một tập hợp (được gọi là “rừng”) các cây quyết định được tạo ra trong quá trình huấn luyện. Sau đó, rừng xuất ra lớp mà đa số cây đã dự đoán. Để tránh khớp quá mức, các yếu tố đa dạng bổ sung được đưa vào quá trình thông qua tính ngẫu nhiên. Một phần của tính ngẫu nhiên đến từ việc “bootstrap” mỗi cây quyết định để mỗi cây nhìn thấy một mẫu ngẫu nhiên của tập huấn luyện. Một phần khác của tính ngẫu nhiên đến từ việc chọn ngẫu nhiên các đầu vào mà mỗi cây xem xét tại mỗi lần phân hoạch trong quá trình huấn luyện. Chìa khóa của cách tiếp cận này là sự khác biệt giữa các cây quyết định được xây dựng và hiệu năng của các mô hình cơ sở riêng lẻ. Nhờ đó, một tổ hợp cây quyết định được tạo ra có hiệu năng vượt trội so với từng mô hình đơn lẻ (Baesens, 2014; Breiman, 2001). Một phương pháp tổ hợp khác, mới hơn, gọi là XGBoost, lần đầu được Chen and Guestrin (2016) đề xuất. Phương pháp này xây dựng một tổ hợp cây quyết định bằng cách sử dụng thuật toán tăng cường gradient để tuần tự xây dựng các mô hình bằng cách khớp các bộ học cơ sở cộng tính nhằm tối thiểu hóa hàm mất mát được cung cấp. Hàm mất mát đo lường mức độ mô hình khớp với dữ liệu, và quá trình tăng cường cũng như bổ sung bộ học cơ sở tiếp tục cho đến khi mức giảm mất mát trở nên tối thiểu. So với các thuật toán tăng cường gradient nói chung, XGBoost thực hiện khai triển Taylor bậc hai cho hàm mục tiêu và sử dụng đạo hàm bậc hai để tăng tốc độ hội tụ của mô hình trong quá trình huấn luyện. Hơn nữa, một số hạng phạt được thêm vào hàm mục tiêu để kiểm soát cấu trúc của mô hình nhằm tránh vấn đề khớp quá mức đã thảo luận ở trên (để biết thêm chi tiết, xem ví dụ Chen and Guestrin (2016); He et al. (2018); Xia et al. (2017)).

### 3.3. Học sâu cho chấm điểm tín dụng

Mạng nơ-ron nhân tạo (ANN) là các mạng gồm những phần tử xử lý đơn giản gọi là nơ-ron. Nơ-ron là các đơn vị tính toán đơn giản nhận một số lượng tùy ý các đầu vào có trọng số (tùy chọn bao gồm một đầu vào thiên lệch) và có thể trả về một đầu ra duy nhất thông qua một hàm kích hoạt. Ý tưởng về nơ-ron có thể được khái quát hóa thành mạng nơ-ron perceptron đa tầng (MLP) bằng cách thêm nhiều tầng chứa nhiều nơ-ron vào mạng này, trong đó mỗi nơ-ron xử lý các đầu vào của nó và tạo ra một giá trị đầu ra được truyền tới tất cả nơ-ron ở tầng tiếp theo. Cấu trúc cơ bản của mạng nơ-ron perceptron đa tầng có một tầng ẩn và một tầng đầu ra. Để tính đầu ra của một nơ-ron ẩn i, trong

$$
h_i=f^{(1)}\left(b_i^{(1)}+\sum_{j=1}^{m}W_{ij}x_j\right)
\tag{2}
$$

trong đó W là ma trận trọng số và Wij biểu thị trọng số nối đầu vào j với đơn vị ẩn i. Tương tự,

$$
y=f^{(2)}\left(b^{(2)}+\sum_{j=1}^{m_h}v_jh_j\right)
\tag{3}
$$

trong đó mh biểu thị số lượng nơ-ron ẩn và v là vectơ trọng số, còn vj là trọng số nối đơn vị ẩn j với nơ-ron đầu ra. Cuối cùng, mạng có khả năng mô hình hóa các quan hệ phi tuyến trong dữ liệu bằng cách sử dụng các hàm kích hoạt f (1) và f (2). Hai loại hàm kích hoạt được sử dụng phổ biến nhất trong mạng nơ-ron là hàm sigmoid và hàm tuyến tính chỉnh lưu (Svozil et al., 1997; Schmidhuber, 2015; Baesens, 2014; Baesens et al., 2003; Spanoudes and Nguyen, 2017). Các tiến bộ trong nghiên cứu mạng nơ-ron và sự gia tăng hiệu năng tính toán đã dẫn đến sự ra đời của các mạng lớn với nhiều tầng ẩn, tức học sâu. Hệ quả là mạng có thể lan truyền trọng số qua mạng và do đó có thể học các độ phức tạp trong các bộ dữ liệu lớn thông qua việc sử dụng nhiều tầng xử lý với cấu trúc phức tạp. Một ví dụ về mạng MLP sâu được thể hiện trong Hình 1. Mạng này được xây dựng bằng nhiều tầng nơ-ron được kết nối với các hàm kích hoạt đơn giản. Khi xử lý các bài toán phân loại, hàm softmax có thể được sử dụng làm hàm kích hoạt trên các nơ-ron nằm ở tầng đầu ra. Để đưa ra dự đoán, mạng sử dụng lớp của tầng đầu ra mà nơ-ron trả về xác suất cao nhất làm lớp dự đoán của mạng. Khi đó, sự khác biệt giữa vectơ xác suất do tầng đầu ra trả về và vectơ nhãn thật có thể được định lượng như một lỗi. Lượng lỗi quyết định cách các trọng số sẽ được điều chỉnh trong quá trình huấn luyện mạng (Spanoudes and Nguyen, 2017; Luo et al., 2017; Van-Sang and Ha-Nam, 2016; Svozil et al., 1997).

Mạng niềm tin sâu (DBN) là một kiến trúc khác, phức tạp hơn, dựa trên mạng nơ-ron. Một ví dụ về loại mạng này được trình bày trong Hình 2. Mạng niềm tin sâu được xây dựng bằng cách sử dụng một số tầng máy Boltzmann hạn chế (RBM), được huấn luyện độc lập với nhau nhằm mã hóa các phụ thuộc thống kê của các đơn vị nằm ở tầng trước. RBM là một đồ thị hai phía, trong đó các đơn vị khả kiến biểu diễn các quan sát được nối với các nơ-ron trong tầng ẩn. Các nơ-ron ẩn này học cách biểu diễn đặc trưng bằng các kết nối có trọng số. RBM bị hạn chế theo nghĩa là không có kết nối khả kiến-với-khả kiến hoặc ẩn-với-ẩn, tương tự thiết lập MLP phân tầng

![Hình 1. Một mạng nơ-ron perceptron đa tầng sâu.](assets/figure-1.png)

*Hình 1. Một mạng nơ-ron perceptron đa tầng sâu. Tầng đầu tiên là tầng đầu vào, tầng cuối cùng là tầng đầu ra, và các tầng ở giữa là các tầng ẩn.*

đã trình bày trước đó (Luo et al., 2017; Mohamed et al., 2012; Lopes and Ribeiro, 2015). Một ví dụ về RBM được thể hiện trong Hình 3. Mạng được học từng tầng một bằng cách sử dụng giá trị của các nơ-ron trong một tầng, khi chúng được suy luận từ dữ liệu, làm đầu vào để huấn luyện tầng tiếp theo. Mục tiêu của mạng là tối đa hóa hợp lý của dữ liệu huấn luyện. Vì vậy, quá trình huấn luyện bắt đầu ở RBM mức thấp nhất, nơi các trạng thái của tầng thấp nhất biểu diễn vectơ dữ liệu đầu vào. Khi các trọng số của RBM mức thấp nhất đã được học, các vectơ kích hoạt đặc trưng ẩn đã học có thể được sử dụng làm dữ liệu để học tầng ẩn thứ hai, và cứ tiếp tục như vậy cho đến cuối cùng RBM ở tầng cuối, chứa các đầu ra của mạng niềm tin sâu, được huấn luyện. Bằng cách thực hiện chuỗi thao tác này, ta thu được một mẫu không chệch của loại vectơ giá trị khả kiến mà mạng tin tưởng (Luo et al., 2017; Hinton and Salakhutdinov, 2006; Mohamed et al., 2011; Lopes and Ribeiro, 2015; Hua et al., 2015).

<table><tr><td><img src="assets/figure-2.png" alt="Hình 2. Mạng niềm tin sâu"></td><td><img src="assets/figure-3.png" alt="Hình 3. Máy Boltzmann hạn chế"></td></tr></table>

*Hình 2. Mạng niềm tin sâu. Mạng được học dần bằng cách xem giá trị của các đơn vị trong một tầng là đầu vào để huấn luyện tầng tiếp theo.*

*Hình 3. Máy Boltzmann hạn chế. Các đơn vị khả kiến biểu diễn các quan sát được kết nối với các đơn vị ẩn, những đơn vị này học các biểu diễn đặc trưng thông qua các kết nối có trọng số.*

Quá trình huấn luyện của mạng niềm tin sâu về bản chất là không giám sát, vì mỗi tầng học các phụ thuộc thống kê từ các đầu vào được cung cấp. Tuy nhiên, mô hình có thể dễ dàng được chuyển thành mô hình phân loại (tức học có giám sát) bằng cách thêm một tầng bổ sung, tương ứng với kích hoạt nhãn lớp, vào mạng niềm tin sâu ban đầu. Khi đó, tầng được thêm vào chỉ cần tinh chỉnh các bộ phát hiện đặc trưng hiện có, vốn đã được phát hiện bởi giai đoạn huấn luyện không giám sát, bằng lan truyền ngược. Mạng truyền thẳng kết quả thường cũng được ký hiệu bằng cùng thuật ngữ DBN (để biết thêm chi tiết về DBN và RBM, xem ví dụ Mohamed et al. (2011, 2009); Lopes and Ribeiro (2015); Hinton (2012); Goodfellow et al. (2016); Vinyals and Ravuri (2011); Mohamed et al. (2012)).

## 4. Thiết lập thực nghiệm

### 4.1. Các mô hình được xây dựng

Hầu hết các phương pháp được thảo luận trong mục trước đều có các siêu tham số cần được người dùng chỉ định. Để bảo đảm thu được các ước lượng tốt về hiệu năng của từng bộ phân loại, một phép tìm kiếm lưới đã được thực hiện để tìm các giá trị tối ưu của siêu tham số cho từng thuật toán phân loại. Các giá trị trong tìm kiếm lưới được lấy từ cả khuyến nghị trong văn liệu và quá trình thăm dò. Các thuật toán khác nhau và lưới tinh chỉnh siêu tham số của chúng được trình bày trong Bảng 2. Trong nỗ lực tăng tốc gradient descent, các mạng MLP được huấn luyện bằng thuật toán tối ưu hóa RMSProp (xem ví dụ Tieleman and Hinton (2012)). Ngoài ra, chuẩn hóa theo lô (xem ví dụ Ioffe and Szegedy (2015)) được đưa vào như một siêu tham số trong lưới tinh chỉnh dùng để huấn luyện mô hình. Hơn nữa, nhằm chống lại xu hướng các mạng nơ-ron sâu khớp quá mức dữ liệu huấn luyện, cả dropout và chính quy hóa L2 (xem ví dụ Goodfellow et al. (2016)) đều được xem xét. Ngoài ra, các mạng sâu với số lượng nơ-ron tăng dần theo từng tầng khó có khả năng khái quát hóa tốt nên không được xem xét. Như thể hiện trong bảng, số lượng mô hình được xây dựng cho các thuật toán dựa trên mạng nơ-ron tăng theo cấp số nhân với số tầng ẩn được sử dụng. Việc xây dựng một nghiệm mạng nơ-ron thường đòi hỏi thời gian và nỗ lực đáng kể, trong đó nhiều mô hình với các cấu hình khác nhau phải được huấn luyện để thu được một nghiệm tốt. Đây có thể được xem là một hạn chế của các mô hình này (Lopes and Ribeiro, 2015).

Chúng tôi đánh giá các bộ phân loại khác nhau trên mười bộ dữ liệu chấm điểm tín dụng bán lẻ nhằm có được một chỉ báo tốt về tính áp dụng tổng quát của các bộ phân loại đối với chấm điểm tín dụng. Bốn bộ dữ liệu, Bene1, Bene2, Bene3 và UK, được thu thập từ các định chế tài chính lớn ở Benelux và Vương quốc Anh. Các bộ dữ liệu Bene1, Bene2 và UK đã từng được sử dụng trong Baesens et al. (2003) và Lessmann et al. (2015). Bộ dữ liệu tín dụng Đức

**Bảng 2. Các thuật toán phân loại được đưa vào thiết lập thực nghiệm và các lưới tinh chỉnh siêu tham số của chúng.**

| Họ | Thuật toán | Số mô hình trên mỗi thuật toán | Siêu tham số | Thiết lập ứng viên |
|---|---|---:|---|---|
| Truyền thống | Logistic regression (LR) | 1 | - | - |
| Truyền thống | Decision tree C4.5 (DT) | 36 | Ngưỡng tin cậy để cắt tỉa | 0.01, 0.1, ..., 0.5 |
| Truyền thống | Decision tree C4.5 (DT) | 36 | Kích thước lá tối thiểu | 3, 4, ..., 8 |
| Tổ hợp | Random forest (RF) | 30 | Số cây CART | 100, 250, 500, 750, 1000 |
| Tổ hợp | Random forest (RF) | 30 | Số đầu vào được lấy mẫu ngẫu nhiên | $\sqrt{m}[0.1, 0.25, 0.5, 1, 2, 4]$ |
| Tổ hợp | XGBoost (XGB) | 108 | Số cây CART | 50, 100, 150 |
| Tổ hợp | XGBoost (XGB) | 108 | Độ sâu cây tối đa | 1, 2, 3 |
| Tổ hợp | XGBoost (XGB) | 108 | Tốc độ học | 0.3, 0.4 |
| Tổ hợp | XGBoost (XGB) | 108 | Tỷ lệ đầu vào được lấy mẫu | 0.6, 0.8 |
| Tổ hợp | XGBoost (XGB) | 108 | Tỷ lệ hàng được lấy mẫu | 0.5, 0.75, 1.00 |
| MLP (lưới chung) | MLP | - | Đơn vị ẩn | 5, 10, 15, 20 |
| MLP (lưới chung) | MLP | - | Tỷ lệ dropout | 0.00, 0.25, 0.50 |
| MLP (lưới chung) | MLP | - | L2 | 0.1, 0.01, 0.001, 0 |
| MLP (lưới chung) | MLP | - | Chuẩn hóa theo lô | Có, Không |
| MLP | MLP1, một tầng ẩn | 144 | Tốc độ học | 1e-2, 1e-3, 1e-4 |
| MLP | MLP3, ba tầng ẩn | 720 | Tốc độ học | 1e-2, 1e-3, 1e-4 |
| MLP | MLP5, năm tầng ẩn | 2016 | Tốc độ học | 1e-3, 1e-4, 1e-5 |
| DBN (lưới chung) | DBN | - | Đơn vị ẩn | 5, 10, 15, 20 |
| DBN (lưới chung) | DBN | - | Dropout tầng ẩn | 0.00, 0.25, 0.50 |
| DBN (lưới chung) | DBN | - | Dropout tầng khả kiến | 0.00, 0.25, 0.50 |
| DBN (lưới chung) | DBN | - | Số bước tương phản phân kỳ | 1, 3, 5 |
| DBN | DBN1, một tầng ẩn | 324 | Tốc độ học | 1.5, 1.0, 0.8 |
| DBN | DBN3, ba tầng ẩn | 1620 | Tốc độ học | 1.0, 0.8, 0.5 |
| DBN | DBN5, năm tầng ẩn | 4536 | Tốc độ học | 1.0, 0.8, 0.5 |

$ m = \lfloor \log_2(N)+1 \rfloor $, theo khuyến nghị của Baesens (2014).

(GC), tín dụng Úc (AC) và tín dụng Đài Loan (TC) có sẵn tại kho UCI (Dua and Graff, 2017; Yeh and Lien, 2009). Bộ dữ liệu GMC được một định chế tài chính cung cấp cho cuộc thi Kaggle “Give me some credit”1. Cuối cùng, bộ dữ liệu TH02 ban đầu được Thomas et al. (2002) sử dụng và bộ dữ liệu HMEQ được lấy từ Baesens et al. (2016). Một số thông tin liên quan về đặc điểm của các bộ dữ liệu được trình bày trong Bảng 3. Trước khi xây dựng các bộ phân loại, các tập đặc trưng của bộ dữ liệu được rút gọn dựa trên hệ số phóng đại phương sai (VIF) để khắc phục các vấn đề liên quan đến đa cộng tuyến, trong đó VIF ≤10 được xem là chấp nhận được. Các tỷ lệ vỡ nợ tiên nghiệm được nêu trong bảng

> Ghi chú: Xem: https://www.kaggle.com/c/GiveMeSomeCredit

cho biết tỷ lệ khoản vay xấu trong các bộ dữ liệu. Vấn đề mất cân bằng lớp đã trở thành một vấn đề lớn trong mô hình hóa dự đoán. Vấn đề này xảy ra khi một trong hai lớp có nhiều trường hợp hơn lớp còn lại. Trong tình huống như vậy, nhiều bộ phân loại sẽ bị thiên lệch về phía lớp đa số và do đó thể hiện hiệu năng phân loại rất kém đối với lớp thiểu số, vốn thường là lớp được quan tâm nhiều hơn, tức vỡ nợ khoản vay. Tuy nhiên, các phương pháp cân bằng lớp sẽ không được sử dụng ở đây vì các lý do do Lessmann et al. (2015) lập luận. Quan trọng nhất, mục tiêu chính của nghiên cứu này là khảo sát sự khác biệt hiệu năng tương đối của một số bộ phân loại, không phải mức hiệu năng tuyệt đối của chúng. Nếu mất cân bằng lớp ảnh hưởng như nhau đến tất cả bộ phân loại, nó ảnh hưởng đến mức hiệu năng tuyệt đối của chúng. Tuy nhiên, nếu một số bộ phân loại được xem xét ít bị ảnh hưởng bởi mất cân bằng lớp hơn, thì đó là lợi ích của việc sử dụng bộ phân loại đó và không nên bị bỏ qua trong phép so sánh của chúng tôi (Lessmann et al., 2015; Zhang et al., 2014).

**Bảng 3. Thông tin về các bộ dữ liệu được đưa vào thiết lập thực nghiệm.**

| Bộ dữ liệu | Số trường hợp | Đầu vào | Tỷ lệ vỡ nợ tiên nghiệm | Kiểm định chéo N×2 |
|---|---:|---:|---:|---:|
| AC | 690 | 14 | 0.445 | 10 |
| GC | 1,000 | 20 | 0.300 | 10 |
| TH02 | 1,225 | 14 | 0.264 | 10 |
| Bene1 | 3,123 | 27 | 0.667 | 10 |
| Bene3 | 3,450 | 8 | 0.016 | 10 |
| HMEQ | 5,960 | 12 | 0.199 | 5 |
| Bene2 | 7,190 | 26 | 0.300 | 5 |
| UK | 30,000 | 14 | 0.040 | 5 |
| TC | 30,000 | 23 | 0.221 | 5 |
| GMC | 150,000 | 10 | 0.067 | 5 |

### 4.2. Tiền xử lý

Chúng tôi áp dụng một cách tiếp cận tiền xử lý chuẩn, trong đó trước hết các giá trị thiếu được điền bằng thay thế trung bình/mốt tương ứng cho các đầu vào số/danh nghĩa. Sau đó, tất cả giá trị của các đầu vào danh nghĩa được thay bằng log của tỷ lệ odds tốt:xấu hoặc trọng số bằng chứng (WOE) cho đầu vào đó (để biết thêm thông tin về WOE, xem ví dụ Baesens (2014); Jiang et al. (2019)). Một quyết định then chốt khi đánh giá các mô hình dự đoán liên quan đến việc xác định phần dữ liệu nào sẽ được dùng để đo hiệu năng của mô hình. Ở đây, kiểm định chéo gấp N×2 được sử dụng; phương pháp này đã được chứng minh là cho kết quả vững hơn so với việc sử dụng một tập huấn luyện và kiểm tra cố định, đặc biệt khi làm việc với bộ dữ liệu nhỏ. Vì vậy, N được đặt tùy theo kích thước của bộ dữ liệu dùng để xây dựng bộ phân loại như trình bày trong Bảng 3. Ngoài ra, như đã thảo luận trước đó trong mục này, nhiều bộ phân loại được xây dựng ở đây phụ thuộc vào các siêu tham số cần được người dùng chỉ định. Để có được các ước lượng tốt về hiệu năng của từng bộ phân loại, một phép tìm kiếm lưới trên các giá trị khả dĩ của các siêu tham số này được thực hiện. Vì vậy, một kiểm định chéo năm gấp bổ sung được thực hiện trong mỗi vòng kiểm định chéo N×2. Mô hình phân loại được chọn ở giai đoạn này sau đó đi vào phép so sánh thực tế, bảo đảm rằng mô hình tốt nhất từ từng thuật toán phân loại khác nhau được so sánh trong vòng kiểm định chéo N×2 bên ngoài (Lessmann et al., 2015; Baesens, 2014).

### 4.3. Chỉ báo hiệu năng

#### 4.3.1. Chỉ báo hiệu năng tiêu chuẩn

Như đã thảo luận trong mục trước, nghiên cứu chuẩn của Lessmann et al. (2015) khuyến nghị rằng các nghiên cứu tương lai nên sử dụng ít nhất ba thước đo hiệu năng để đánh giá mô hình chấm điểm tín dụng, cụ thể là AUC, Gini từng phần và Brier Score, vì các thước đo này vừa phổ biến trong chấm điểm tín dụng vừa đo các khía cạnh khác nhau của hiệu năng bộ phân loại. AUC đánh giá khả năng phân biệt của một bộ phân loại bằng cách đo diện tích dưới đường cong ROC và bằng xác suất rằng một người vỡ nợ được chọn ngẫu nhiên sẽ nhận điểm cao hơn một người không vỡ nợ được chọn ngẫu nhiên (Lessmann et al., 2015). AUC đưa ra đánh giá toàn cục về hiệu năng của một bộ phân loại, vì nó xét toàn bộ phân phối điểm. Do đó, nó giả định rằng mọi ngưỡng đều có khả năng xảy ra như nhau. Giả định này không hoàn toàn thực tế trong chấm điểm tín dụng, vì bên cho vay chỉ chấp nhận các hồ sơ có điểm thấp hơn một ngưỡng xác định. Vì lý do này, độ chính xác của bộ phân loại ở vùng thấp của phân phối điểm có tầm quan trọng đặc biệt. Một thước đo hiệu năng khác, Gini từng phần, tập trung vào khả năng phân biệt của bộ phân loại trong phần của phân phối điểm nằm dưới ngưỡng xác định $p(+1\mid x)$ ≤b. Ở đây b sẽ được chọn bằng 0.4, như Lessmann et al. (2015) đề xuất. Cuối cùng, Brier score đánh giá độ chính xác của dự đoán xác suất bằng cách tính sai số bình phương trung bình giữa $p(+1\mid x)$ và biến phản hồi nhị phân (Bradley, 1997; Lessmann et al., 2015; Thomas et al., 2002).

#### 4.3.2. Thước đo phân loại dựa trên lợi nhuận

Các thước đo hiệu năng truyền thống như những thước đo đã thảo luận ở mục trước thường không thể tính đến chính xác thực tế kinh doanh của chấm điểm tín dụng. Thước đo lợi nhuận kỳ vọng tối đa (EMP) được phát triển để phù hợp hơn với các mối quan tâm kinh doanh, và là một thước đo hiệu năng tổng quát ước lượng lợi nhuận mà một công ty có thể đạt được bằng cách áp dụng một bộ phân loại cụ thể (Verbraken et al., 2014). Lợi nhuận phân loại trung bình cho mỗi khách hàng đạt được bằng cách sử dụng một bộ phân loại cho mô hình hóa PD như mô tả ở trên được tính như sau:

$$
P(t;b_1,c_0,c^*)=(b_1-c^*)\pi_1F_1(t)-(c_0+c^*)\pi_0F_0(t)
\tag{4}
$$

trong đó b1 là lợi ích của việc phân loại đúng một khách hàng sẽ vỡ nợ, c0 là chi phí của việc phân loại một khách hàng không vỡ nợ thành người vỡ nợ, và c∗ là chi phí chung của một hành động do công ty thực hiện. Xác suất tiên nghiệm của vỡ nợ (không vỡ nợ) là π1 (π0) và F1(t) (F0(t)) là hàm mật độ tích lũy của vỡ nợ (không vỡ nợ) cho trước ngưỡng t. Lợi nhuận trung bình ước lượng là một hàm của ngưỡng t; tối ưu hóa hàm này dẫn tới thước đo lợi nhuận tối đa (MP) được định nghĩa là MP = maxP,∀t(t; b1; c0, c∗). Vì các tham số chi phí và lợi ích, c0 và b1, không phải lúc nào cũng có thể được xác định chính xác từ trước, giá trị kỳ vọng

$$
\mathrm{EMP}=\int_{b_1}\int_{c_0}P\!\left(T(\theta);b_1,c_0,c^*\right)h(b_1,c_0)\,dc_0\,db_1
\tag{5}
$$

với h(b1, c0) là mật độ xác suất chung của chi phí phân loại. Do đó, EMP là thước đo lợi nhuận mà một công ty có thể đạt được bằng cách áp dụng một bộ phân loại. Trên thực tế, đã chứng minh rằng EMP là cận trên của lợi nhuận mà một công ty có thể đạt được khi áp dụng một bộ phân loại cụ thể (để biết thêm chi tiết về thước đo EMP, ví dụ liên quan đến việc xác định các tham số chi phí và lợi ích, xem Verbraken et al. (2014)).

### 4.4. So sánh mô hình

Trong mục tiếp theo, một số phương pháp chấm điểm tín dụng sẽ được đánh giá và so sánh trên mười bộ dữ liệu chấm điểm tín dụng bán lẻ. Trong lịch sử, một số phương pháp đã được đề xuất nhằm so sánh thống kê hiệu năng của nhiều bộ phân loại trên một số bộ dữ liệu. Một cách tiếp cận đã được thiết lập là thực hiện kiểm định Friedman và kiểm định giả thuyết không rằng hiệu năng của các bộ phân loại được so sánh là tương đương. Nếu kiểm định bác bỏ giả thuyết không này, một kiểm định hậu nghiệm có thể được thực hiện để so sánh tất cả bộ phân loại với nhau (Demˇsar, 2006; Garćıa et al., 2010). Trong lịch sử, khái niệm “ý nghĩa thống kê” đã được sử dụng để củng cố các kết luận của phát hiện khoa học và thường được đánh giá bằng một chỉ số gọi là p-value. Mặc dù hữu ích, p-value thường bị sử dụng sai và diễn giải sai. Điều này khiến một số nhà thống kê khuyến cáo hạn chế việc sử dụng chúng; ví dụ, American Statistical Association khuyến nghị rằng các kết luận khoa học không nên chỉ dựa trên việc một p-value có vượt qua một ngưỡng cụ thể hay không. Do đó, các phương pháp này đã mất dần vị thế trong nhiều lĩnh vực khoa học (Wasserstein et al., 2016; Benavoli et al., 2017; Kruschke and Liddell, 2018). Dưới đây, một vài hạn chế nền tảng của các phương pháp NHST tần suất (kiểm định thống kê giả thuyết không) sẽ được thảo luận.

Trong học máy, các nhà nghiên cứu thường xây dựng một số phương pháp trên một tập hợp các bộ dữ liệu và cố gắng chứng minh rằng một phương pháp vượt trội hơn các phương pháp khác. Để xác nhận kết quả, một NHST tần suất được thực hiện và kết quả được xem là có ý nghĩa ở mức tin cậy 95% (α = 0.05) nếu p-value ≤0.05. Trong trường hợp này, giả thuyết mà chúng ta quan tâm là xác suất rằng hiệu năng của các phương pháp được xem xét là khác nhau (hoặc bằng nhau). Vì vậy, chúng ta quan tâm đến việc biết tính khả dĩ của giả thuyết không (tức không có khác biệt về hiệu năng trung bình giữa các phương pháp) cho trước dữ liệu, $p(H_0\mid D)$. Tuy nhiên, các phương pháp NHST tần suất không thể trả lời câu hỏi này một cách thỏa đáng. Trên thực tế, chúng cung cấp cho chúng ta xác suất thu được dữ liệu của chúng ta (tức khác biệt quan sát được về hiệu năng trung bình giữa các phương pháp), với điều kiện giả thuyết không là đúng, tức $p(D\mid H_0)$. Chúng ta hành xử như thể α bằng tỷ lệ các trường hợp trong đó $H_0$ sẽ bị bác bỏ sai nếu chúng ta lặp lại thí nghiệm. Điều này sẽ đúng nếu p-value cho chúng ta xác suất của giả thuyết. Tuy nhiên, p-value cung cấp cho chúng ta xác suất của dữ liệu của chúng ta, hoặc dữ liệu chưa quan sát cực đoan hơn, với điều kiện giả thuyết không của chúng ta là đúng. Do đó, p-value tóm tắt dữ liệu khi giả định một giả thuyết không cụ thể, nhưng nó không thể đi ngược lại và đưa ra phát biểu về thực tại nền tảng (Wasserstein et al., 2016; Kruschke and Liddell, 2018; Kruschke et al., 2012; Benavoli et al., 2017; Nuzzo, 2014). Hơn nữa, giả thuyết không mà các phương pháp này kiểm định phát biểu rằng hiệu năng của các bộ phân loại là bằng nhau. Trong thực tế, giả thuyết này hầu như luôn sai, vì không có hai bộ phân loại nào có hiệu năng hoàn toàn tương đương. Nếu một phương pháp NHST bác bỏ giả thuyết không của chúng ta, điều đó cho thấy giả thuyết là khó xảy ra; tuy nhiên điều này đã được biết trước khi thí nghiệm được thực hiện. Đây là một vấn đề của kiểm định giả thuyết không theo trường phái tần suất nói chung, vì hầu hết các yếu tố quan tâm đều có một quan hệ khác không nào đó, ngay cả khi hiệu ứng rất nhỏ. Trong học máy, điều này dẫn đến việc giả thuyết không có thể bị bác bỏ bằng cách kiểm thử các bộ phân loại cạnh tranh trên đủ dữ liệu, vì cỡ mẫu có thể được nhà nghiên cứu quyết định. Một hệ quả khác là các khác biệt có thể hình dung được có thể không dẫn đến p-value nhỏ nếu mẫu được sử dụng không đủ lớn. Trong những thập kỷ qua, p-value thường được hiểu như một chỉ báo về kích thước hiệu ứng. Tuy nhiên, trong thực tế nó là một hàm của cả kích thước hiệu ứng lẫn cỡ mẫu, và do đó các p-value giống nhau không hàm ý các kích thước hiệu ứng giống nhau. Vì vậy, p-value và rộng hơn là ý nghĩa thống kê không đo lường hoặc chỉ ra kích thước của một hiệu ứng hay tầm quan trọng của một kết quả. Do đó, ý nghĩa thống kê không tương đương với ý nghĩa thực tiễn (Kruschke and Liddell, 2018; Benavoli et al., 2017; Lesaffre and Lawson, 2012; Wasserstein et al., 2016). Cuối cùng, các phương pháp NHST tần suất không cung cấp thông tin nào về giả thuyết không; do đó, khi giả thuyết không không bị bác bỏ thì không thể đưa ra kết luận. Một p-value lớn gợi ý rằng dữ liệu không bất thường nếu giả thuyết không là đúng. Trong nhiều trường hợp, điều này chỉ đơn thuần cho thấy dữ liệu không thể phân biệt giữa nhiều giả thuyết cạnh tranh. Vì vậy, diễn giải một kết quả không có ý nghĩa như bằng chứng ủng hộ giả thuyết không (chẳng hạn không có khác biệt hiệu năng giữa hai bộ phân loại) là sai, vì các phương pháp NHST tần suất không thể cung cấp bằng chứng ủng hộ giả thuyết không (Kruschke, 2011; Kruschke and Liddell, 2018; Greenland et al., 2016; Benavoli et al., 2017). Thảo luận này minh họa một số hạn chế của các phương pháp NHST tần suất. Tuy nhiên, nó không nhằm đưa ra một tổng quan đầy đủ về các hạn chế của các phương pháp này (để thảo luận thêm, xem ví dụ Greenland et al. (2016); Benavoli et al. (2017); Nuzzo (2014)). Động lực của thảo luận ở trên là nhấn mạnh một số hạn chế nền tảng của các phương pháp NHST tần suất. Cốt lõi của những hạn chế này là thực tế rằng các phương pháp này không cung cấp cho chúng ta câu trả lời cho các câu hỏi mà chúng ta thực sự quan tâm, tức xác suất của giả thuyết của chúng ta cho trước dữ liệu thực tế. Tuy nhiên, đây chính xác là điều được cung cấp bởi phân phối hậu nghiệm trong suy luận thống kê Bayes. Do đó, một cách để khắc phục nhiều hạn chế của NHST tần suất là chuyển sang kiểm định giả thuyết Bayes (Kruschke and Liddell, 2018; Benavoli et al., 2017; Corani et al., 2017). Một phương pháp gần đây như vậy, do Benavoli et al. (2014) phát triển, có thể được dùng để so sánh các bộ phân loại trên nhiều bộ dữ liệu. Đây là đối ứng Bayes của kiểm định hạng có dấu theo trường phái tần suất và tính đến vùng tương đương thực tiễn (ROPE), có thể được sử dụng như một quy tắc quyết định tinh vi hơn so với α, vì nó cũng cung cấp một cách để chấp nhận giả thuyết không thay vì chỉ bác bỏ nó. Phương pháp nhận đầu vào là một vectơ chứa khác biệt hiệu năng trên một số bộ dữ liệu. Khi được trang bị ROPE, kiểm định hạng có dấu Bayes sau đó có thể tính các xác suất hậu nghiệm rằng hai bộ phân loại là tương đương về mặt thực tiễn hoặc khác biệt đáng kể (Benavoli et al., 2014, 2017; Kruschke et al., 2012; Corani et al., 2017).

## 5. Kết quả

Trong mục này, hiệu năng của các bộ phân loại được xem xét sẽ được so sánh. Như đã thảo luận trong mục trước, tất cả bộ phân loại được xem xét đã được so sánh trên mười bộ dữ liệu chấm điểm tín dụng bán lẻ và được đánh giá bằng bốn thước đo hiệu năng nhằm thu được một chỉ báo tốt về tính áp dụng tổng quát của các bộ phân loại cho chấm điểm tín dụng. Sau đây, hai thủ tục kiểm định thống kê sẽ được thực hiện để so sánh mô hình. Trước hết, hiệu năng của các bộ phân loại được xem xét sẽ được phân tích bằng các phương pháp tần suất, vốn là quy ước trong chấm điểm tín dụng. Sau đó, phân tích thống kê Bayes sẽ được tiến hành. Phép so sánh này sẽ củng cố các phát hiện thực nghiệm và làm sáng tỏ tính áp dụng của các thủ tục kiểm định thống kê này cho chấm điểm tín dụng.

### 5.1. So sánh mô hình theo trường phái tần suất

Cơ sở cho phân tích thống kê tần suất là các thứ hạng trung bình của các bộ phân loại như trình bày trong Bảng 4. Ở đây, các bộ phân loại được xếp hạng trên các bộ dữ liệu và chỉ báo độ chính xác, trong đó bộ phân loại có hiệu năng tốt nhất đối với một thước đo hiệu năng và bộ dữ liệu nhất định được xếp hạng một, còn bộ phân loại tệ nhất nhận hạng mười. Hơn nữa, thứ hạng trung bình của mỗi bộ phân loại trên tất cả thước đo hiệu năng được trình bày trong bảng dưới cột Avg. Hàng cuối cùng của Bảng 4 trình bày thống kê kiểm định và p-value của kiểm định Friedman. Kiểm định này so sánh các thứ hạng trung bình của tất cả bộ phân loại đối với các thước đo hiệu năng được xem xét và kiểm định giả thuyết không rằng thứ hạng của các bộ phân loại là bằng nhau (Lessmann et al., 2015; Garćıa et al., 2010; Demˇsar, 2006). Như có thể thấy từ bảng, giả thuyết không này bị bác bỏ đối với tất cả thước đo hiệu năng (p <.000). Tiếp theo, một phép so sánh từng cặp được thực hiện, trong đó tất cả bộ phân loại được so sánh với bộ phân loại có hiệu năng tốt nhất theo từng thước đo hiệu năng và thủ tục Rom được dùng để bù trừ cho kiểm định nhiều lần (Demˇsar, 2006). Các p-value thu được từ các phép so sánh này được trình bày trong ngoặc ở Bảng 4, trong đó gạch dưới chỉ ra rằng giả thuyết không về việc một bộ phân loại có hiệu năng tốt ngang với bộ phân loại tốt nhất đã bị bác bỏ (tức p <.05).

**Bảng 4. Thứ hạng trung bình của các bộ phân loại trên các bộ dữ liệu theo từng thước đo hiệu năng.** Các giá trị trong ngoặc là p-value đã hiệu chỉnh từ phép so sánh với bộ phân loại tốt nhất. Cột cuối cùng là thứ hạng trung bình.

| Bộ phân loại | AUC | Brier score | Gini từng phần | EMP | Trung bình |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 4.1 (.078) | 3.5 (.069) | 4.3 (.182) | 3.8 (.237) | 3.9 |
| Decision Tree | 6.7 (.000) | 6.3 (.000) | 7.0 (.000) | 6.3 (.000) | 6.6 |
| Random Forest | 2.9 (.369) | 3.6 (.069) | 3.2 (.700) | 3.1 (.335) | 3.2 |
| **XGBoost** | **2.4** | **1.8** | **3.0** | **2.6** | **2.5** |
| MLP, 1 tầng ẩn | 3.1 (.369) | 3.0 (.095) | 3.1 (.700) | 3.3 (.335) | 3.1 |
| MLP, 3 tầng ẩn | 3.4 (.307) | 5.0 (.001) | 3.4 (.700) | 3.6 (.276) | 3.9 |
| MLP, 5 tầng ẩn | 5.8 (.000) | 4.8 (.001) | 4.3 (.182) | 5.6 (.001) | 5.1 |
| DBN, 1 tầng ẩn | 8.6 (.000) | 8.6 (.000) | 8.8 (.000) | 8.6 (.000) | 8.7 |
| DBN, 3 tầng ẩn | 8.9 (.000) | 9.0 (.000) | 8.8 (.000) | 9.0 (.000) | 8.9 |
| DBN, 5 tầng ẩn | 9.1 (.000) | 9.4 (.000) | 9.1 (.000) | 9.1 (.000) | 9.2 |
| Friedman $\chi^2_9$ | 70.0 (.000) | 72.0 (.000) | 67.2 (.000) | 66.7 (.000) | - |

Nhiều hiểu biết mới có thể thu được từ phép so sánh này. Thứ nhất, phương pháp tổ hợp XGBoost có thứ hạng tổng thể tốt nhất trong tất cả bộ phân loại được xem xét ở đây. Cụ thể hơn, bộ phân loại này có thứ hạng trung bình tốt nhất trên các bộ dữ liệu dựa trên tất cả thước đo hiệu năng được xem xét. Thứ hai, hai phương pháp truyền thống cho chấm điểm tín dụng có hiệu năng kém hơn bộ phân loại tốt nhất trên tất cả thước đo hiệu năng được xem xét. Cây quyết định có thứ hạng trung bình 6.6 và hoạt động kém hơn đáng kể so với bộ phân loại tốt nhất trên tất cả thước đo hiệu năng. Hồi quy logistic có thứ hạng trung bình cao hơn và không xác định được khác biệt có ý nghĩa khi bộ phân loại này được so sánh với bộ phân loại có hiệu năng tốt nhất theo từng thước đo hiệu năng. Như đã thảo luận trong mục trước, điều này không thể được diễn giải là bằng chứng ủng hộ giả thuyết không về hiệu năng bằng nhau vì các phương pháp NHST tần suất không cung cấp thông tin về giả thuyết không. Do đó, không thể rút ra kết luận khi giả thuyết không không bị bác bỏ. Thứ ba, một MLP với năm tầng ẩn hoạt động kém hơn đáng kể so với XGBoost dựa trên các thước đo AUC, Brier score và EMP. Cùng kiến trúc với ba tầng ẩn hoạt động kém hơn đáng kể so với XGBoost dựa trên Brier score. Một mạng MLP nông với một tầng ẩn là bộ phân loại có hiệu năng tốt thứ hai dựa trên thứ hạng tổng thể của nó. Cụ thể hơn, mạng này là bộ phân loại có hiệu năng tốt thứ hai dựa trên tất cả thước đo hiệu năng được xem xét ở đây ngoại trừ EMP, nơi Random Forest là bộ phân loại có hiệu năng tốt thứ hai. Cuối cùng, ba bộ phân loại có hiệu năng tệ nhất dựa trên thứ hạng trung bình là các DBN với một, ba và năm tầng ẩn; tất cả đều hoạt động kém hơn đáng kể so với bộ phân loại tốt nhất trên tất cả thước đo hiệu năng được xem xét ở đây.

![Hình 4. Khác biệt từng cặp trong hiệu năng của các mạng MLP và DBN.](assets/figure-4.png)

*Hình 4. Khác biệt từng cặp trong hiệu năng của các mạng MLP và DBN; các ô màu xám biểu thị khác biệt có ý nghĩa ở mức tin cậy 95%.*

Động lực chính của nghiên cứu này là khảo sát mức độ phù hợp của các thuật toán học sâu đối với chấm điểm tín dụng. Từ phần trên, có thể kết luận rằng phương pháp tổ hợp XGBoost nhìn chung vượt trội hơn các mạng này, cho thấy nên ưu tiên phương pháp này cho chấm điểm tín dụng nếu hiệu năng phân loại hoặc EMP là mục tiêu của hoạt động chấm điểm tín dụng. Cũng đáng quan tâm khi khảo sát thêm cách các mạng sâu hoạt động so với các đối ứng nông hơn của chúng. Từ việc quan sát thứ hạng trung bình của các bộ phân loại trong Bảng 4, có thể thấy các mạng sâu dường như không cải thiện hiệu năng so với các đối ứng nông hơn. Trên thực tế, hiệu năng của các mạng dường như giảm dần khi độ phức tạp của chúng tăng lên. Để khảo sát thêm hiệu năng của các mạng sâu với nhiều tầng ẩn so với các đối ứng nông hơn, các khác biệt từng cặp giữa tất cả bộ phân loại kiểu mạng nơ-ron được khám phá bằng thủ tục so sánh bội Nemenyi (Hollander et al., 2014). Kết quả được trình bày trong Hình 4, trong đó các ô màu xám biểu thị những khác biệt có ý nghĩa thống kê. Như có thể thấy từ các hình, không có mạng sâu nào có hiệu năng khác biệt đáng kể so với đối ứng nông hơn của nó.

### 5.2. So sánh mô hình Bayes

Phân tích tần suất như phân tích được thực hiện ở trên có một số hạn chế nền tảng như đã mô tả trong Mục 4.4. Đáng chú ý nhất là nó không cung cấp cho chúng ta xác suất của giả thuyết mà thực ra chúng ta quan tâm kiểm định. Các thủ tục kiểm định thống kê Bayes có thể được triển khai để khắc phục các hạn chế của phương pháp NHST tần suất. Kiểm định hạng có dấu Bayes dựa trên các khác biệt hiệu năng của các bộ phân loại được xem xét trên một số bộ dữ liệu. Ví dụ, hiệu năng của các bộ phân loại dựa trên AUC trên một số bộ dữ liệu được trình bày trong Hình 5. Một số hiểu biết thú vị có thể được rút ra từ hình này mà không hiển nhiên ngay khi quan sát các thứ hạng trung bình của các bộ phân loại như trình bày trong Bảng 4.

![Hình 5. Hiệu năng bộ phân loại theo AUC.](assets/figure-5.png)

*Hình 5. Hiệu năng của các bộ phân loại khác nhau trên tất cả bộ dữ liệu được xem xét, được đo bằng AUC.*

Thứ nhất, bộ phân loại có hiệu năng tổng thể tốt nhất, XGBoost, là bộ phân loại có hiệu năng tốt nhất đối với năm trong số mười bộ dữ liệu được xem xét. Thứ hai, ba DBN hoạt động kém hơn đáng kể so với các bộ phân loại khác được xem xét trên tất cả bộ dữ liệu được xem xét. Hơn nữa, mặc dù chuẩn ngành, hồi quy logistic, nhìn chung hoạt động hợp lý, các phương pháp tiên tiến hơn hoạt động tốt hơn đáng kể trên một số bộ dữ liệu. Một ví dụ là bộ dữ liệu HMEQ, nơi hiệu năng của các phương pháp tổ hợp và một MLP với ba tầng ẩn tốt hơn đáng kể so với hiệu năng của mô hình hồi quy logistic. Điều này có thể là một dấu hiệu của các hiệu ứng phi tuyến và/hoặc tương tác không được mô hình hồi quy logistic nắm bắt. Thước đo EMP cung cấp một góc nhìn có động cơ kinh tế về hiệu năng của các bộ phân loại được xem xét. Nó đo lợi nhuận gia tăng phát sinh từ việc sử dụng một bộ phân loại nhất định so với một kịch bản cơ sở trong đó tất cả khoản vay đều được cấp, được biểu thị theo tỷ lệ phần trăm của tổng số tiền vay (Verbraken et al., 2014). Kịch bản cơ sở được sử dụng bảo đảm tính nhất quán khi đánh giá một số mô hình chấm điểm tín dụng; tuy nhiên, khả năng sinh lợi ước lượng phát sinh từ việc sử dụng một mô hình chấm điểm tín dụng phụ thuộc vào số lượng người vỡ nợ trong bộ dữ liệu. Ba trong số mười bộ dữ liệu được xem xét ở đây bị mất cân bằng nghiêm trọng, cụ thể là các bộ dữ liệu UK, Bene3 và GMC. Điều này hiển nhiên khi quan sát Hình 6. Như trước, XGBoost là bộ phân loại có hiệu năng tốt nhất thường xuyên nhất. Ngoài ra, các DBN là các bộ phân loại có hiệu năng tệ nhất trên tất cả bộ dữ liệu được xem xét. Như có thể thấy, không thu được lợi nhuận gia tăng nào khi các mạng được đánh giá trên ba bộ dữ liệu mất cân bằng nghiêm trọng, cho thấy các DBN không thể xác định chính xác các

![Hình 6. Hiệu năng bộ phân loại theo EMP.](assets/figure-6.png)

*Hình 6. Hiệu năng của các bộ phân loại khác nhau trên tất cả bộ dữ liệu được xem xét, được đo bằng EMP. Khả năng sinh lợi ước lượng phụ thuộc vào số lượng người vỡ nợ; các bộ dữ liệu mất cân bằng nghiêm trọng được hiển thị ở bên trái.*

Khi thực hiện phân tích Bayes, thí nghiệm được tóm tắt bằng phân phối hậu nghiệm. Bằng cách truy vấn phân phối này, có thể đánh giá xác suất của giả thuyết. Chẳng hạn, chúng ta có thể suy luận xác suất rằng một bộ phân loại có hiệu năng thực tiễn tốt hơn hoặc kém hơn một bộ phân loại khác, hoặc liệu chúng có tương đương về mặt thực tiễn hay không. Để làm điều này, kích thước của ROPE phải được xác định. Trong phân tích của chúng tôi, ROPE được đặt bằng 0.01 khi đánh giá kết quả dựa trên AUC và Gini từng phần. Sau đó, thủ tục thống kê Bayes tính xem bao nhiêu phần của phân phối hậu nghiệm kết quả của khác biệt trung bình nằm trong ROPE, là khoảng (-0.01, 0.01). Khi đánh giá kết quả dựa trên Brier score và EMP, ROPE được đặt ở các giá trị thấp hơn, lần lượt là 0.0025 và 0.001, được xem là phù hợp hơn với hai thước đo hiệu năng này. Kết quả của một thủ tục thống kê Bayes có thể được trực quan hóa bằng một đơn hình xác suất như trong Hình 7. Trong hình, hai bộ phân loại (XGBoost và hồi quy logistic) được so sánh dựa trên AUC bằng so sánh thống kê Bayes. Hình cho thấy các mẫu từ các hậu nghiệm (đám mây điểm) và ba vùng của phân phối hậu nghiệm. Vùng ở góc dưới bên trái biểu diễn trường hợp XGBoost có xác suất lớn hơn cả hồi quy logistic và ROPE gộp lại; vùng ở đỉnh tam giác tương ứng với trường hợp ROPE có xác suất lớn hơn hai bộ phân loại gộp lại; và cuối cùng, vùng ở góc dưới bên phải biểu diễn trường hợp hồi quy logistic có xác suất lớn hơn XGBoost và ROPE gộp lại. Dựa trên hình, chúng ta có thể thấy một tỷ lệ lớn các trường hợp ủng hộ XGBoost. Điều này có thể được định lượng bằng số bằng cách tính tỷ lệ điểm rơi vào ba vùng. Khi làm như vậy, chúng tôi thấy XGBoost tốt hơn trong 57.8% trường hợp và do đó có thể kết luận với xác suất 57.8% rằng XGBoost có hiệu năng thực tiễn tốt hơn hồi quy logistic. Điều này minh họa lợi ích của các thủ tục kiểm định thống kê Bayes được xây dựng bằng phân phối hậu nghiệm có trang bị ROPE. Các thủ tục này cho phép chúng ta ước lượng xác suất hậu nghiệm của một giả thuyết không hợp lý, tức diện tích bên trong ROPE, và khẳng định các khác biệt có ý nghĩa thực tiễn, tức diện tích bên ngoài ROPE (Benavoli et al., 2017).

![Hình 7. So sánh thống kê Bayes giữa XGBoost và hồi quy logistic.](assets/figure-7.png)

*Hình 7. So sánh thống kê Bayes giữa XGBoost và hồi quy logistic dựa trên AUC, cho thấy các mẫu hậu nghiệm và phân bố của chúng giữa ba vùng.*

Kết quả của phân tích Bayes, trong đó tất cả bộ phân loại được so sánh với bộ phân loại có hiệu năng tốt nhất đối với từng thước đo hiệu năng trên tất cả bộ dữ liệu, được trình bày trong Bảng 5. Mỗi ô trong bảng chứa ba giá trị chỉ tỷ lệ các trường hợp rơi vào từng vùng trong ba vùng của đơn hình xác suất: giá trị phía dưới bên trái là xác suất hậu nghiệm rằng bộ phân loại có hiệu năng tốt nhất đối với từng thước đo hiệu năng có hiệu năng thực tiễn tốt hơn bộ phân loại được nêu ở cột đầu; giá trị phía dưới bên phải là xác suất hậu nghiệm rằng bộ phân loại được nêu ở cột đầu có hiệu năng thực tiễn tốt hơn bộ phân loại có hiệu năng tốt nhất; và giá trị phía trên là xác suất hậu nghiệm của ROPE, cho thấy các bộ phân loại được so sánh là tương đương về mặt thực tiễn. Kết quả thu được bằng phân tích Bayes cung cấp cho chúng ta các xác suất của những quyết định mà chúng ta thực sự quan tâm. Do đó, chúng ta có thể ra quyết định bằng các xác suất này, vốn có thể được diễn giải trực tiếp, trái ngược với p-value. Chẳng hạn, chúng ta có thể quyết định như một quy tắc cứng rằng một kết quả có ý nghĩa thu được nếu một trong ba xác suất vượt ngưỡng 95%. Điều này được biểu thị bằng gạch dưới trong Bảng 5. Cần lưu ý ở đây rằng việc sử dụng một quy tắc cứng để khẳng định một kết quả có ý nghĩa thống kê, mặc dù thường hữu ích khi muốn thực hiện nhiều phân tích, lại đưa vào tư duy đen trắng có những hạn chế. Một cách để khắc phục điều này là tính odds hậu nghiệm khi không có xác suất hậu nghiệm nào đạt ngưỡng. Ví dụ, có thể tính odds hậu nghiệm của việc XGBoost có hiệu năng thực tiễn tốt hơn hồi quy logistic bằng cách tính o(XG, LR) = p(XG)/p(LR). Tính odds dựa trên AUC cho kết quả 288, cho thấy bằng chứng mạnh ủng hộ XGBoost dựa trên thước đo hiệu năng này (Benavoli et al., 2017; Corani et al., 2017).

Sử dụng các phương pháp NHST tần suất, không xác định được khác biệt hiệu năng có ý nghĩa khi chuẩn ngành, hồi quy logistic, được so sánh với bộ phân loại có hiệu năng tốt nhất, XGBoost. Như đã thảo luận trước đó, điều này không thể được diễn giải là bằng chứng về việc không có khác biệt, vì các phương pháp NHST tần suất không thể cung cấp bằng chứng ủng hộ giả thuyết không. Khi quan sát kết quả của phân tích Bayes trong đó hai bộ phân loại được so sánh, chúng ta có thể thấy không có xác suất hậu nghiệm thu được nào vượt quá quy tắc cứng 95%. Tuy nhiên, như trình bày trong bảng, phần lớn các trường hợp ủng hộ XGBoost so với hồi quy logistic dựa trên AUC (57.6%) và Gini từng phần (83.8%). Nếu chúng ta tính odds hậu nghiệm dựa trên kết quả của hai thước đo hiệu năng này, có thể kết luận rằng có bằng chứng mạnh cho thấy XGBoost vượt trội thực tiễn so với hồi quy logistic dựa trên AUC (o(XGB, LR) = 288) và bằng chứng tích cực cho thấy XGBoost vượt trội thực tiễn so với hồi quy logistic dựa trên Gini từng phần (o(XGB, LR) = 7). Như trước, cây quyết định hoạt động kém hơn đáng kể so với XGBoost dựa trên tất cả thước đo hiệu năng được xem xét với xác suất từ 97.7-100%. Không thể rút ra kết luận khi hai phương pháp tổ hợp được so sánh bằng các phương pháp NHST tần suất. Khi quan sát kết quả của phân tích Bayes, có thể thấy rõ rằng không có xác suất hậu nghiệm thu được nào vượt quá

**Bảng 5. So sánh Bayes với bộ phân loại có thứ hạng tốt nhất (XGBoost) theo từng thước đo hiệu năng.** Mỗi ô được trình bày theo dạng `ROPE / XGBoost tốt hơn / bộ phân loại của hàng tốt hơn`.

| Bộ phân loại | AUC | Brier score | Gini từng phần | EMP |
|---|---|---|---|---|
| Logistic Regression | .404 / .576 / .002 | .467 / .377 / .156 | .042 / .838 / .120 | .656 / .266 / .078 |
| Decision Tree | .001 / .999 / .000 | .000 / 1.000 / .000 | .000 / 1.000 / .000 | .023 / .977 / .000 |
| Random Forest | .886 / .099 / .015 | .852 / .110 / .038 | .026 / .549 / .426 | .945 / .000 / .055 |
| **Thứ hạng trung bình của XGBoost** | **2.4** | **1.8** | **3.0** | **2.6** |
| MLP, 1 tầng ẩn | .734 / .264 / .002 | .518 / .366 / .117 | .111 / .481 / .408 | .928 / .071 / .000 |
| MLP, 3 tầng ẩn | .530 / .467 / .003 | .145 / .855 / .000 | .010 / .670 / .319 | .794 / .201 / .005 |
| MLP, 5 tầng ẩn | .013 / .987 / .000 | .094 / .906 / .000 | .001 / .733 / .266 | .462 / .538 / .000 |
| DBN, 1 tầng ẩn | .000 / 1.000 / .000 | .000 / 1.000 / .000 | .000 / 1.000 / .000 | .001 / .999 / .000 |
| DBN, 3 tầng ẩn | .000 / 1.000 / .000 | .000 / 1.000 / .000 | .000 / 1.000 / .000 | .001 / .999 / .000 |
| DBN, 5 tầng ẩn | .000 / 1.000 / .000 | .000 / 1.000 / .000 | .000 / 1.000 / .000 | .001 / .999 / .000 |

quy tắc cứng 95%. Tuy nhiên, một phần lớn của phân phối hậu nghiệm (85.2-94.5%) dường như ủng hộ ROPE đối với ba trong bốn thước đo hiệu năng được xem xét, gợi ý rằng hiệu năng của hai bộ phân loại là tương đương về mặt thực tiễn dựa trên các thước đo hiệu năng đó. Như trước, ba kiến trúc DBN được xem xét ở đây hoạt động kém hơn đáng kể so với XGBoost dựa trên tất cả thước đo hiệu năng được xem xét với xác suất từ 99.9-100%. Mạng MLP với năm tầng ẩn cũng hoạt động kém hơn đáng kể so với XGBoost dựa trên AUC với xác suất 98.7%. Cùng kiến trúc với ba tầng ẩn được phát hiện hoạt động kém hơn đáng kể so với XGBoost dựa trên Brier score bằng các phương pháp NHST tần suất. Khi các bộ phân loại được so sánh bằng các phương pháp Bayes, không có xác suất hậu nghiệm thu được nào vượt quá quy tắc cứng 95%. Tuy nhiên, một phần lớn các trường hợp ủng hộ XGBoost dựa trên Brier score (85.5%). Ngoài ra, có bằng chứng mạnh ủng hộ ROPE so với kiến trúc MLP (o(ROPE, MLP3) = 158) và bằng chứng tích cực ủng hộ ROPE so với XGBoost (o(ROPE, XGB) = 4) dựa trên EMP, cho thấy hiệu năng của hai bộ phân loại là tương đương về mặt thực tiễn dựa trên thước đo hiệu năng đó. Cuối cùng, không xác định được khác biệt có ý nghĩa thống kê khi hiệu năng của mạng MLP với một tầng ẩn được so sánh với hiệu năng của bộ phân loại tốt nhất bằng các phương pháp tần suất. Khi quan sát kết quả từ so sánh mô hình Bayes, chúng ta có thể thấy một phần đáng kể của phân phối hậu nghiệm dường như ủng hộ ROPE dựa trên ba trong bốn thước đo hiệu năng được xem xét. Ví dụ, phần lớn các trường hợp được lấy mẫu từ hậu nghiệm ủng hộ ROPE dựa trên EMP (92.8%), cho thấy hiệu năng của các bộ phân loại là tương đương về mặt thực tiễn dựa trên thước đo hiệu năng đó.

**Bảng 6. So sánh Bayes giữa các bộ phân loại mạng nơ-ron.** Mỗi ô được trình bày theo dạng `ROPE / bộ phân loại bên trái tốt hơn / bộ phân loại bên phải tốt hơn`.

| So sánh | AUC | Brier score | Gini từng phần | EMP |
|---|---|---|---|---|
| DBN-1 vs. DBN-3 | .964 / .000 / .036 | 1.000 / .000 / .000 | .078 / .282 / .640 | 1.000 / .000 / .000 |
| DBN-1 vs. DBN-5 | .927 / .011 / .062 | 1.000 / .000 / .000 | .060 / .290 / .650 | 1.000 / .000 / .000 |
| DBN-3 vs. DBN-5 | 1.000 / .000 / .000 | 1.000 / .000 / .000 | .552 / .365 / .083 | 1.000 / .000 / .000 |
| MLP-1 vs. MLP-3 | .963 / .000 / .037 | .999 / .001 / .000 | .097 / .223 / .680 | .963 / .000 / .037 |
| MLP-1 vs. MLP-5 | .124 / .876 / .000 | .960 / .040 / .000 | .016 / .787 / .197 | .558 / .442 / .000 |
| MLP-3 vs. MLP-5 | .301 / .699 / .000 | 1.000 / .000 / .000 | .099 / .705 / .196 | .461 / .539 / .000 |

Để khảo sát cách các mạng sâu so sánh với các đối ứng nông hơn của chúng, các khác biệt từng cặp về hiệu năng của các bộ phân loại dựa trên mạng nơ-ron được khám phá bằng các thủ tục kiểm định thống kê Bayes như trình bày trong Bảng 6. Khi phân tích bằng các phương pháp NHST tần suất, không tìm thấy khác biệt có ý nghĩa nào khi hiệu năng của các mạng sâu được so sánh với hiệu năng của đối ứng nông hơn; cần nhấn mạnh lại ở đây rằng điều này không thể được diễn giải là bằng chứng về việc không có khác biệt trong hiệu năng của các phương pháp, vì không thể rút ra kết luận khi các phương pháp NHST tần suất không bác bỏ giả thuyết không. Tuy nhiên, điều này có thể được kiểm định chính thức bằng các thủ tục kiểm định thống kê Bayes. Như có thể thấy từ Bảng 6, các mạng sâu nhìn chung dường như không cải thiện hiệu năng so với các đối ứng nông hơn. Khi xét các DBN, hiệu năng của mạng với ba tầng ẩn là tương đương về mặt thực tiễn với đối ứng một tầng của nó dựa trên AUC, Brier score và EMP. Hiệu năng của cùng mạng với năm tầng ẩn cũng tương đương về mặt thực tiễn với DBN có một tầng ẩn dựa trên Brier score và EMP. Hơn nữa, hiệu năng của DBN với năm tầng ẩn là tương đương về mặt thực tiễn với hiệu năng của cùng mạng với ba tầng ẩn dựa trên AUC, Brier score và EMP. Từ việc quan sát các kết quả so sánh đối với các mạng MLP, các kết luận tương tự xuất hiện. Một MLP với một tầng ẩn duy nhất có hiệu năng tương đương về mặt thực tiễn với cùng mạng có ba tầng ẩn dựa trên AUC, Brier score và EMP. Mạng với một tầng ẩn cũng có hiệu năng tương đương về mặt thực tiễn với cùng mạng có năm tầng ẩn dựa trên Brier score. Ngoài ra, một MLP với ba tầng ẩn có hiệu năng tương đương về mặt thực tiễn với cùng mạng có năm tầng ẩn dựa trên Brier score.

## 6. Kết luận

Mức độ phù hợp của các thuật toán phân loại khác nhau cho chấm điểm tín dụng đã được nghiên cứu rộng rãi kể từ khi lĩnh vực này hình thành vào thập niên 1950. Trong thập kỷ qua, nghiên cứu về chấm điểm tín dụng đã tính đến sự xuất hiện của các phương pháp tổ hợp và kết luận rằng một phương pháp như vậy, rừng ngẫu nhiên, nên được xem là phương pháp chuẩn cho chấm điểm tín dụng. Gần đây hơn, XGBoost đã được đề xuất và xem xét cho chấm điểm tín dụng, trong đó nó đã được chứng minh là vượt trội hơn rừng ngẫu nhiên trong một số trường hợp. Tuy nhiên, nghiên cứu về các thuật toán phân loại cho chấm điểm tín dụng phần lớn đã bỏ qua sự phát triển của các kiến trúc học sâu. Điều này đòi hỏi một cập nhật tiếp theo cho nghiên cứu bằng cách xem xét các thuật toán học sâu cho chấm điểm tín dụng. Để đạt mục tiêu đó, hai kiến trúc học sâu đã được xây dựng, cụ thể là mạng niềm tin sâu và mạng perceptron đa tầng, rồi được so sánh với các phương pháp truyền thống cho chấm điểm tín dụng, hồi quy logistic và cây quyết định, cùng hai phương pháp tổ hợp cho chấm điểm tín dụng, rừng ngẫu nhiên và XGBoost. Các bộ phân loại khác nhau được so sánh dựa trên bốn chỉ báo hiệu năng trên mười bộ dữ liệu. Cuối cùng, các thủ tục kiểm định thống kê Bayes được giới thiệu trong bối cảnh chấm điểm tín dụng và được so sánh với các phương pháp NHST tần suất, vốn theo truyền thống được xem là thực hành tốt nhất trong chấm điểm tín dụng. Phép so sánh này nhấn mạnh nhiều lợi ích của các thủ tục kiểm định thống kê Bayes và củng cố các phát hiện thực nghiệm.

Chủ yếu có thể rút ra hai kết luận từ việc so sánh các bộ phân loại khác nhau. Thứ nhất, XGBoost là bộ phân loại có thứ hạng tổng thể tốt nhất trong tất cả bộ phân loại được xem xét ở đây và là bộ phân loại có hiệu năng tốt nhất dựa trên tất cả thước đo hiệu năng được xem xét. Thứ hai, các mạng sâu với một số tầng ẩn, tức học sâu, không vượt trội hơn các mạng nông hơn với một tầng ẩn. Cũng cần tính đến trong phép so sánh này rằng các mạng sâu đi kèm chi phí tính toán lớn hơn nhiều so với các bộ phân loại khác được xem xét ở đây, vì số lượng mô hình cần xây dựng để tinh chỉnh đầy đủ các siêu tham số của mô hình tăng theo cấp số nhân với số tầng ẩn. Do đó, có thể kết luận rằng các thuật toán học sâu dường như không phải là phương pháp phù hợp cho chấm điểm tín dụng và rằng một phương pháp tổ hợp, XGBoost, nhìn chung nên được ưu tiên hơn các phương pháp chấm điểm tín dụng khác được xem xét ở đây khi hiệu năng phân loại là mục tiêu chính của hoạt động chấm điểm tín dụng.

Như đã thảo luận trước đó, kỷ nguyên hiện đại của nghiên cứu về mạng nơ-ron có thể được quy cho công trình tiên phong của McCulloch and Pitts (1943), những người đã chỉ ra rằng về lý thuyết mạng nơ-ron có thể khớp bất kỳ hàm tính toán được nào. Dưới ánh sáng của kết quả này, có vẻ đáng ngạc nhiên khi một phương pháp tổ hợp, cụ thể là XGBoost, lại là bộ phân loại có hiệu năng tốt nhất và các mô hình dựa trên mạng nơ-ron không thể vượt trội hơn bộ phân loại này. Trong nghiên cứu này, hai kiến trúc dựa trên mạng nơ-ron đã được xem xét và xây dựng với một, ba và năm tầng ẩn. Từ phân tích của chúng tôi, có thể kết luận rằng nhìn chung các mạng sâu không vượt trội hơn các đối ứng nông hơn của chúng. Cụ thể hơn, nghiên cứu phát hiện rằng trong nhiều trường hợp, hiệu năng của các mạng sâu thường gần như bằng về mặt thực tiễn với hiệu năng của đối ứng một tầng nông hơn. Một giải thích hợp lý cho việc các mạng sâu không vượt trội hơn các phương pháp khác được xem xét ở đây có khả năng đến từ thực tế rằng học sâu đã được chứng minh là rất giỏi trong việc phát hiện các cấu trúc phức tạp, với điều kiện có nhiều thể hiện để học từ đó (LeCun et al., 2015), điều có thể không đúng đối với hầu hết các bộ dữ liệu rủi ro tín dụng.

Cũng cần lưu ý rằng các mô hình tổ hợp là những mô hình được gọi là “hộp đen”, cho thấy khó diễn giải vì sao các mô hình này đạt một kết quả nhất định hoặc đưa ra một dự đoán nhất định. Nếu khả năng diễn giải dự đoán của mô hình là mối quan tâm chính, người ta có thể muốn quay lại các phương pháp truyền thống cho chấm điểm tín dụng, ví dụ hồi quy logistic. Tuy nhiên, nếu hiệu năng dự đoán là trọng tâm chính của việc xây dựng mô hình, thì nhìn chung XGBoost dường như là lựa chọn tốt nhất. Hơn nữa, các mô hình dựa trên mạng nơ-ron được xem xét ở đây, tức mạng perceptron đa tầng và mạng niềm tin sâu, cũng được xem là các mô hình hộp đen. Do đó, nhìn chung các mô hình này có hiệu năng dự đoán cho chấm điểm tín dụng kém hơn hai phương pháp tổ hợp được xem xét ở đây và khó xây dựng cũng như diễn giải hơn. Đáng lưu ý là ngay cả khi khả năng diễn giải là điều kiện tiên quyết cho hoạt động chấm điểm tín dụng, các mô hình hộp đen, vốn đã cho thấy hiệu năng tốt trong chấm điểm tín dụng, vẫn là công cụ chuẩn đối sánh có giá trị. Điều này là do các mô hình này có thể được sử dụng để xác định các hiệu ứng phi tuyến quan trọng và/hoặc các tương tác trong bộ dữ liệu chấm điểm tín dụng. Nếu các phi tuyến tính có hệ quả được xác định, thì người thực hành có thể xấp xỉ chúng bằng các mô hình cộng tổng quát. Khung xây dựng mô hình này đã được đề xuất và áp dụng thành công trong văn liệu (ví dụ Van Gestel et al. (2005, 2006)). Hơn nữa, trong những thập kỷ qua, lĩnh vực AI có thể giải thích (XAI) đã xuất hiện, tập trung vào việc phát triển các phương pháp để diễn giải dự đoán của các mô hình hộp đen (Adadi and Berrada, 2018). Điều này đã dẫn đến sự phát triển của một số phương pháp cho mục đích này (xem ví dụ Ribeiro et al. (2016); Lundberg and Lee (2017)).

Để đạt được kết quả của chúng tôi, một số lượng đáng kể các bộ dữ liệu đời thực đã được đưa vào thiết lập thực nghiệm. Các bộ dữ liệu này khá đa dạng cả về số lượng quan sát lẫn đầu vào được sử dụng; do đó nghiên cứu này cung cấp một chỉ báo tốt về hiệu năng tổng quát của các bộ phân loại được xem xét cho chấm điểm tín dụng. Một hướng thú vị cho công việc tương lai là xem xét các nguồn dữ liệu ít truyền thống hơn cho chấm điểm tín dụng, ví dụ các nguồn dữ liệu phi cấu trúc như hình ảnh hoặc văn bản, nhằm tăng cường độ phong phú của các nguồn dữ liệu được xem xét và tiếp tục cải thiện hiệu năng của các thuật toán phân loại cho chấm điểm tín dụng. Chẳng hạn, việc sử dụng dữ liệu điện thoại di động và dữ liệu văn bản đã cho thấy các kết quả rất hứa hẹn về mặt này (ví dụ ́Oskarsd́ottir et al. (2019); Stevenson et al. (2020)). Ngoài ra, nghiên cứu tương lai có thể mở rộng công trình này bằng cách tính đến các thuật toán học sâu mới khác như mạng nơ-ron tích chập.

## Tài liệu tham khảo

- Adadi, A. and Berrada, M. (2018). Peeking inside the black-box: A survey on explainable artificial intelligence (xai). IEEE Access, 6:52138–52160.

- Addo, P., Guegan, D., and Hassani, B. (2018). Credit risk analysis using machine and deep learning models. Risks, 6(2):38.

- Akkoç, S. (2012). An empirical comparison of conventional techniques, neural networks and the three stage hybrid adaptive neuro fuzzy inference system (anfis) model for credit scoring analysis: The case of turkish credit card data. European Journal of Operational Research, 222(1):168–178.

- Baesens, B. (2014). Analytics in a big data world: The essential guide to data science and its applications. John Wiley & Sons.

- Baesens, B., Roesch, D., and Scheule, H. (2016). Credit Risk Analytics: Measurement Techniques, Applications, and Examples in SAS. John Wiley & Sons.

- Baesens, B., Van Gestel, T., Viaene, S., Stepanova, M., Suykens, J., and Vanthienen, J. (2003). Benchmarking state-of-the-art classification algorithms for credit scoring. Journal of the operational research society, 54(6):627–635.

- Benavoli, A., Corani, G., Demšar, J., and Zaffalon, M. (2017). Time for a change: a tutorial for comparing multiple classifiers through bayesian analysis. The Journal of Machine Learning Research, 18(1):2653–2688.

- Benavoli, A., Corani, G., Mangili, F., Zaffalon, M., and Ruggeri, F. (2014). A bayesian wilcoxon signed-rank test based on the dirichlet process. In International conference on machine learning, pages 1026–1034.

- Board of Governors of the Federal Reserve System (2019). Federal reserve statistical release. https://www.federalreserve. gov/releases/h8/current/default.htm. [Online; accessed 28-February-2019].

- Bradley, A. P. (1997). The use of the area under the roc curve in the evaluation of machine learning algorithms. Pattern recognition, 30(7):1145–1159.

- Breiman, L. (2001). Random forests. Machine learning, 45(1):5–32.

- Chen, S., Guo, Z., and Zhao, X. (2020). Predicting mortgage early delinquency with machine learning methods. European Journal of Operational Research.

- Chen, T. and Guestrin, C. (2016). Xgboost: A scalable tree boosting system. In Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining, pages 785–794. ACM.

- Corani, G., Benavoli, A., Demšar, J., Mangili, F., and Zaffalon, M. (2017). Statistical comparison of classifiers through bayesian hierarchical modelling. Machine Learning, 106(11):1817–1837.

- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. Journal of Machine learning research, 7(Jan):1–30.

- Deng, L. (2014). A tutorial survey of architectures, algorithms, and applications for deep learning. APSIPA Transactions on Signal and Information Processing, 3.

- Dua, D. and Graff, C. (2017). UCI machine learning repository.

- Durand, D. (1941). Risk elements in consumer installment financing. National Bureau of Economic Research, New York.

- Garcı́a, S., Fernández, A., Luengo, J., and Herrera, F. (2010). Advanced nonparametric tests for multiple comparisons in the design of experiments in computational intelligence and data mining: Experimental analysis of power. Information Sciences, 180(10):2044–2064.

- Goodfellow, I., Bengio, Y., and Courville, A. (2016). Deep learning. MIT press.

- Greenland, S., Senn, S. J., Rothman, K. J., Carlin, J. B., Poole, C., Goodman, S. N., and Altman, D. G. (2016). Statistical tests, p values, confidence intervals, and power: a guide to misinterpretations. European journal of epidemiology, 31(4):337–350.

- Hamori, S., Kawai, M., Kume, T., Murakami, Y., and Watanabe, C. (2018). Ensemble learning or deep learning? application to default risk analysis. Journal of Risk and Financial Management, 11(1):12.

- Haykin, S. (1994). Neural networks, volume 2. Prentice hall New York.

- He, H., Zhang, W., and Zhang, S. (2018). A novel ensemble method for credit scoring: Adaption of different imbalance ratios. Expert Systems with Applications, 98:105–117.

- Hinton, G. E. (2012). A practical guide to training restricted boltzmann machines. In Neural networks: Tricks of the trade, pages 599–619. Springer.

- Hinton, G. E. and Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks. Science, 313(5786):504– 507.

- Hollander, M., Wolfe, D. A., and Chicken, E. (2014). Nonparametric statistical methods, volume 751. John Wiley & Sons.

- Hosmer Jr, D. W., Lemeshow, S., and Sturdivant, R. X. (2013). Applied logistic regression, volume 398. John Wiley & Sons.

- Hssina, B., Merbouha, A., Ezzikouri, H., and Erritali, M. (2014). A comparative study of decision tree id3 and c4. 5. International Journal of Advanced Computer Science and Applications, 4(2).

- Hua, Y., Guo, J., and Zhao, H. (2015). Deep belief networks and deep learning. In Intelligent Computing and Internet of Things (ICIT), 2014 International Conference on, pages 1–4. IEEE.

- Huang, Y.-M., Hung, C.-M., and Jiau, H. C. (2006). Evaluation of neural networks and data mining methods on a credit assessment task for class imbalance problem. Nonlinear Analysis: Real World Applications, 7(4):720–747.

- Ioffe, S. and Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167.

- Jiang, C., Wang, Z., and Zhao, H. (2019). A prediction-driven mixture cure model and its application in credit scoring. European Journal of Operational Research, 277(1):20–31.

- Kraus, M., Feuerriegel, S., and Oztekin, A. (2020). Deep learning in business analytics and operations research: Models, applications and managerial implications. European Journal of Operational Research, 281(3):628–641.

- Kruschke, J. K. (2011). Doing bayesian data analysis: A tutorial with r and bugs. burlington, ma.

- Kruschke, J. K., Aguinis, H., and Joo, H. (2012). The time has come: Bayesian methods for data analysis in the organizational sciences. Organizational Research Methods, 15(4):722–752.

- Kruschke, J. K. and Liddell, T. M. (2018). The bayesian new statistics: Hypothesis testing, estimation, meta-analysis, and power analysis from a bayesian perspective. Psychonomic Bulletin & Review, 25(1):178–206.

- LeCun, Y., Bengio, Y., and Hinton, G. (2015). Deep learning. nature, 521(7553):436.

- Lesaffre, E. and Lawson, A. B. (2012). Bayesian biostatistics. John Wiley & Sons.

- Lessmann, S., Baesens, B., Seow, H.-V., and Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. European Journal of Operational Research, 247(1):124–136.

- Lopes, N. and Ribeiro, B. (2015). Machine Learning for Adaptive Many-Core Machines-A Practical Approach. Springer.

- Lundberg, S. M. and Lee, S.-I. (2017). A unified approach to interpreting model predictions. In Advances in neural information processing systems, pages 4765–4774.

- Luo, C., Wu, D., and Wu, D. (2017). A deep learning approach for credit scoring using credit default swaps. Engineering Applications of Artificial Intelligence, 65:465–470.

- Maldonado, S., Bravo, C., López, J., and Pérez, J. (2017). Integrated framework for profit-based feature selection and svm classification in credit scoring. Decision Support Systems, 104:113–121.

- Mancisidor, R. A., Kampffmeyer, M., Aas, K., and Jenssen, R. (2019). Deep generative models for reject inference in credit scoring. arXiv preprint arXiv:1904.11376.

- Marqués, A., Garcı́a, V., and Sánchez, J. S. (2012). Two-level classifier ensembles for credit risk assessment. Expert Systems with Applications, 39(12):10916–10922.

- McCulloch, W. S. and Pitts, W. (1943). A logical calculus of the ideas immanent in nervous activity. The bulletin of mathematical biophysics, 5(4):115–133.

- Mohamed, A.-r., Dahl, G., and Hinton, G. (2009). Deep belief networks for phone recognition. In Nips workshop on deep learning for speech recognition and related applications, number 9 in 1, page 39. Vancouver, Canada.

- Mohamed, A.-r., Dahl, G. E., and Hinton, G. (2012). Acoustic modeling using deep belief networks. IEEE Transactions on Audio, Speech, and Language Processing, 20(1):14–22.

- Mohamed, A.-r., Sainath, T. N., Dahl, G. E., Ramabhadran, B., Hinton, G. E., Picheny, M. A., et al. (2011). Deep belief networks using discriminative features for phone recognition. In ICASSP, pages 5060–5063.

- Munkhdalai, L., Wang, L., Park, H. W., and Ryu, K. H. (2019). Advanced neural network approach, its explanation with lime for credit scoring application. In Asian Conference on Intelligent Information and Database Systems, pages 407–419. Springer.

- Nuzzo, R. (2014). Scientific method: statistical errors. Nature News, 506(7487):150.

- Óskarsdóttir, M., Bravo, C., Sarraute, C., Vanthienen, J., and Baesens, B. (2019). The value of big data for credit scoring: Enhancing financial inclusion using mobile phone data and social network analytics. Applied Soft Computing, 74:26–39.

- Papouskova, M. and Hajek, P. (2019). Two-stage consumer credit risk modelling using heterogeneous ensemble learning. Decision Support Systems, 118:33–45.

- Ribeiro, M. T., Singh, S., and Guestrin, C. (2016). ” why should i trust you?” explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining, pages 1135–1144.

- Saberi, M., Mirtalaie, M. S., Hussain, F. K., Azadeh, A., Hussain, O. K., and Ashjari, B. (2013). A granular computing-based approach to credit scoring modeling. Neurocomputing, 122:100–115.

- Schmidhuber, J. (2015). Deep learning in neural networks: An overview. Neural networks, 61:85–117.

- Sharma, S., Agrawal, J., and Sharma, S. (2013). Classification through machine learning technique: C4. 5 algorithm based on various entropies. International Journal of Computer Applications, 82(16).

- Spanoudes, P. and Nguyen, T. (2017). Deep learning in customer churn prediction: Unsupervised feature learning on abstract company independent feature vectors. arXiv preprint arXiv:1703.03869.

- Stevenson, M., Mues, C., and Bravo, C. (2020). The value of text for small business default prediction: A deep learning approach. arXiv preprint arXiv:2003.08964.

- Sun, T. and Vasarhelyi, M. A. (2018). Predicting credit card delinquencies: An application of deep neural networks. Intelligent Systems in Accounting, Finance and Management, 25(4):174–189.

- Svozil, D., Kvasnicka, V., and Pospichal, J. (1997). Introduction to multi-layer feed-forward neural networks. Chemometrics and intelligent laboratory systems, 39(1):43–62.

- Thomas, L. C., Edelman, D. B., and Crook, J. N. (2002). Credit scoring and its applications. SIAM.

- Tieleman, T. and Hinton, G. (2012). Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural networks for machine learning, 4(2):26–31.

- Van Gestel, T., Baesens, B., Van Dijcke, P., Garcia, J., Suykens, J. A., and Vanthienen, J. (2006). A process model to develop an internal rating system: Sovereign credit ratings. Decision Support Systems, 42(2):1131–1151.

- Van Gestel, T., Baesens, B., Van Dijcke, P., Suykens, J., Garcia, J., and Alderweireld, T. (2005). Linear and nonlinear credit scoring by combining logistic regression and support vector machines. Journal of credit Risk, 1(4).

- Van-Sang, H. and Ha-Nam, N. (2016). Credit scoring with a feature selection approach based deep learning. In MATEC Web of Conferences, volume 54. EDP Sciences.

- Verbraken, T., Bravo, C., Weber, R., and Baesens, B. (2014). Development and application of consumer credit scoring models using profit-based classification measures. European Journal of Operational Research, 238(2):505–513.

- Vinyals, O. and Ravuri, S. V. (2011). Comparing multilayer perceptron to deep belief network tandem features for robust asr. In 2011 IEEE international conference on acoustics, speech and signal processing (ICASSP), pages 4596–4599. IEEE.

- Wang, C., Han, D., Liu, Q., and Luo, S. (2018a). A deep learning approach for credit scoring of peer-to-peer lending using attention mechanism lstm. IEEE Access, 7:2161–2168.

- Wang, M., Yu, J., and Ji, Z. (2018b). Personal credit risk assessment based on stacking ensemble model. In International Conference on Intelligent Information Processing, pages 328–333. Springer.

- Wasserstein, R. L., Lazar, N. A., et al. (2016). The asa’s statement on p-values: context, process, and purpose. The American Statistician, 70(2):129–133.

- Xia, Y., Liu, C., Li, Y., and Liu, N. (2017). A boosted decision tree approach using bayesian hyper-parameter optimization for credit scoring. Expert Systems with Applications, 78:225–241.

- Xiao, W., Zhao, Q., and Fei, Q. (2006). A comparative study of data mining methods in consumer loans credit scoring management. Journal of Systems Science and Systems Engineering, 15(4):419–435.

- Yeh, I.-C. and Lien, C.-h. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2):2473–2480.

- Yu, L., Yao, X., Wang, S., and Lai, K. K. (2011). Credit risk evaluation using a weighted least squares svm classifier with design of experiment for parameter selection. Expert Systems with Applications, 38(12):15392–15399.

- Zhang, Z., Gao, G., and Shi, Y. (2014). Credit risk evaluation using multi-criteria optimization classifier with kernel, fuzzification and penalty factors. European Journal of Operational Research, 237(1):335–348.

- Zhou, L., Lai, K. K., and Yu, L. (2010). Least squares support vector machines ensemble models for credit scoring. Expert Systems with Applications, 37(1):127–133.

- Zhu, B., Yang, W., Wang, H., and Yuan, Y. (2018). A hybrid deep learning model for consumer credit scoring. In 2018 International Conference on Artificial Intelligence and Big Data (ICAIBD), pages 205–208. IEEE.
