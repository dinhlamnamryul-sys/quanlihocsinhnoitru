import face_recognition
import pickle
import os

# Giả sử thầy có thư mục chứa ảnh xếp theo phòng: 
# dataset/Phòng 1 (Nam)/Giang_A_Pao.jpg
# dataset/Phòng 1 (Nam)/Lo_Van_Cho.jpg

THU_MOC_ANH = "dataset"
du_lieu_nha_truong = {}

print("Đang tiến hành mã hóa khuôn mặt. Quá trình này có thể mất vài phút...")

for ten_phong in os.listdir(THU_MOC_ANH):
    duong_dan_phong = os.path.join(THU_MOC_ANH, ten_phong)
    if not os.path.isdir(duong_dan_phong):
        continue
        
    du_lieu_nha_truong[ten_phong] = {}
    
    for ten_file in os.listdir(duong_dan_phong):
        if ten_file.endswith((".jpg", ".png", ".jpeg")):
            ten_hoc_sinh = os.path.splitext(ten_file)[0].replace("_", " ")
            duong_dan_anh = os.path.join(duong_dan_phong, ten_file)
            
            # Đọc ảnh và mã hóa khuôn mặt thành chuỗi số 128 chiều
            anh_goc = face_recognition.load_image_file(duong_dan_anh)
            cac_khuon_mat = face_recognition.face_encodings(anh_goc)
            
            if len(cac_khuon_mat) > 0:
                # Lấy khuôn mặt đầu tiên tìm thấy trong ảnh
                du_lieu_nha_truong[ten_phong][ten_hoc_sinh] = cac_khuon_mat[0]
                print(f"✅ Đã mã hóa thành công: {ten_hoc_sinh} ({ten_phong})")
            else:
                print(f"❌ Không tìm thấy khuôn mặt trong ảnh của: {ten_hoc_sinh}")

# Xuất ra file .pkl siêu nhẹ (chỉ khoảng vài MB cho 1000 học sinh)
with open("dulieu_khuanmat.pkl", "wb") as f:
    pickle.dump(du_lieu_nha_truong, f)

print("🎉 Hoàn tất! File dulieu_khuanmat.pkl đã sẵn sàng để đẩy lên Streamlit.")