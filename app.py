import streamlit as st
import face_recognition
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import datetime

# 1. Cấu hình trang web (Ưu tiên hiển thị trên điện thoại)
st.set_page_config(page_title="Nội trú Sam Mứn", page_icon="🏫", layout="centered")

# 2. Tải dữ liệu khuôn mặt đã mã hóa (.pkl)
@st.cache_data
def load_data():
    try:
        with open("dulieu_khuanmat.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Chưa tìm thấy file dulieu_khuanmat.pkl. Hãy chạy file tao_du_lieu.py trước!")
        return {}

du_lieu = load_data()
danh_sach_phong = list(du_lieu.keys()) if du_lieu else ["Chưa có dữ liệu"]

# 3. Giao diện chính
st.title("🏫 Quản lý Nội trú Sam Mứn")
st.caption("Ứng dụng AI điểm danh và giám sát nề nếp")

# Chọn phòng cần điểm danh
phong_chon = st.selectbox("📍 Chọn phòng điểm danh:", danh_sach_phong)

# Nút bật camera quét điểm danh
anh_chup = st.camera_input("📸 Bấm vào đây để quét khuôn mặt phòng")

if anh_chup is not None and du_lieu:
    st.info("Đang dùng AI phân tích ảnh...")
    
    # Chuyển đổi dữ liệu ảnh từ điện thoại sang mảng Numpy cho thuật toán
    img = Image.open(anh_chup)
    img_array = np.array(img)
    
    # Trích xuất đặc trưng các khuôn mặt trong ảnh vừa chụp
    vitri_khuonmat = face_recognition.face_locations(img_array)
    mahoa_khuonmat = face_recognition.face_encodings(img_array, vitri_khuonmat)
    
    # Lấy dữ liệu danh sách học sinh của ĐÚNG phòng được chọn (Tối ưu tốc độ)
    hoc_sinh_trong_phong = du_lieu[phong_chon]
    ten_hs_trong_phong = list(hoc_sinh_trong_phong.keys())
    mahoa_hs_trong_phong = list(hoc_sinh_trong_phong.values())
    
    hs_co_mat = []
    
    # Quá trình so khớp bằng AI (Dung sai 0.5 để nhận diện khắt khe hơn)
    for mahoa in mahoa_khuonmat:
        ket_qua = face_recognition.compare_faces(mahoa_hs_trong_phong, mahoa, tolerance=0.5)
        if True in ket_qua:
            # Lấy người giống nhất nếu có nhiều kết quả True
            khoang_cach = face_recognition.face_distance(mahoa_hs_trong_phong, mahoa)
            vi_tri_tot_nhat = np.argmin(khoang_cach)
            if ket_qua[vi_tri_tot_nhat]:
                ten = ten_hs_trong_phong[vi_tri_tot_nhat]
                hs_co_mat.append(ten)
    
    # Lọc bỏ trùng lặp (trường hợp AI nhận diện 1 người 2 lần trong 1 khung hình)
    hs_co_mat = list(set(hs_co_mat))
    
    # Tính toán vắng mặt
    hs_vang_mat = list(set(ten_hs_trong_phong) - set(hs_co_mat))
    
    # Hiển thị kết quả
    st.divider()
    st.subheader(f"📊 Kết quả: {phong_chon}")
    st.write(f"**Sĩ số:** {len(ten_hs_trong_phong)} | **Có mặt:** {len(hs_co_mat)}")
    
    if len(hs_vang_mat) == 0:
        st.success("✅ Đủ sĩ số. Không có học sinh vắng.")
    else:
        st.error(f"⚠️ Vắng mặt ({len(hs_vang_mat)}): {', '.join(hs_vang_mat)}")
        
    # Xuất file Excel báo cáo (tự động điền ngày giờ hiện tại)
    thoi_gian_thuc = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    df = pd.DataFrame({
        "Thời gian": [thoi_gian_thuc],
        "Phòng": [phong_chon],
        "Sĩ số": [len(ten_hs_trong_phong)],
        "Vắng mặt": [", ".join(hs_vang_mat) if hs_vang_mat else "Đủ"]
    })
    
    # Tải file CSV thi đua
    csv = df.to_csv(index=False).encode('utf-8-sig') # Dùng utf-8-sig để không bị lỗi font tiếng Việt
    st.download_button(
        label="📥 Tải nhật ký điểm danh (Excel/CSV)",
        data=csv,
        file_name=f'diem_danh_{phong_chon}.csv',
        mime='text/csv',
    )

st.divider()

# 4. Tính năng Trợ lý phát thanh song ngữ (Việt - H'Mông)
st.subheader("🔊 Trợ lý nhắc nhở trật tự")
if st.button("🚨 Phát loa cảnh báo phòng ồn ào"):
    try:
        # Thầy nhớ để file nhac_nho.mp3 vào cùng thư mục chứa code
        st.audio("nhac_nho.mp3", format="audio/mp3", autoplay=True)
        st.warning("Đã phát cảnh báo nhắc nhở bằng tiếng Việt và H'Mông.")
    except Exception as e:
        st.error("Chưa tìm thấy file âm thanh nhac_nho.mp3")