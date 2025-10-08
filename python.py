import streamlit as st
import pandas as pd
import numpy as np # Cần để tính NPV và IRR
import json
import time
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Đánh giá Phương án Kinh doanh",
    layout="wide"
)

# Khai báo biến API Key (Tự động lấy từ Streamlit Secrets)
API_KEY = st.secrets.get("GEMINI_API_KEY")

st.title("Ứng dụng Đánh giá Phương án Kinh doanh (CAPEX) 💼")
st.markdown("---")

# --- Hàm tính toán NPV, IRR, PP, DPP (Task 3) ---
def calculate_project_metrics(cash_flows, wacc_rate):
    """Tính toán NPV, IRR, PP, DPP dựa trên dòng tiền và WACC."""
    
    # NPV (Net Present Value - Giá trị hiện tại ròng)
    # np.npv(rate, values)
    npv = np.npv(wacc_rate, cash_flows)
    
    # IRR (Internal Rate of Return - Tỷ suất sinh lời nội bộ)
    # np.irr(values)
    irr = np.irr(cash_flows)
    
    # Tính Payback Period (PP) và Discounted Payback Period (DPP)
    
    # 1. Tính Payback Period (PP - Thời gian hoàn vốn)
    cumulative_cf = np.cumsum(cash_flows)
    payback_period = float('inf') 
    
    # Lặp qua dòng tiền tích lũy để tìm điểm hoàn vốn (cum_cf >= 0)
    for i, cum_cf in enumerate(cumulative_cf):
        if cum_cf >= 0:
            if i == 0:
                 payback_period = 0 
            else:
                 # Số tiền còn lại cần hoàn vốn ngay trước năm i
                 amount_to_recover = -cumulative_cf[i-1] 
                 # Thời gian cần thêm trong năm i: (Số tiền cần bù đắp) / (Dòng tiền ròng năm i)
                 fractional_year = amount_to_recover / cash_flows[i]
                 payback_period = i - 1 + fractional_year
            break
        
    # 2. Tính Discounted Payback Period (DPP - Thời gian hoàn vốn chiết khấu)
    
    # Tính dòng tiền chiết khấu (Discounted Cash Flow - DCF)
    dcf = [cash_flows[i] / ((1 + wacc_rate) ** i) for i in range(len(cash_flows))]
    dcf_cumulative = np.cumsum(dcf)
    
    discounted_payback_period = float('inf') 
    for i, cum_dcf in enumerate(dcf_cumulative):
        if cum_dcf >= 0:
            if i == 0:
                discounted_payback_period = 0
            else:
                 # Số tiền còn lại cần hoàn vốn ngay trước năm i
                 amount_to_recover = -dcf_cumulative[i-1] 
                 # Thời gian cần thêm trong năm i: (Số tiền cần bù đắp) / (DCF năm i)
                 fractional_year = amount_to_recover / dcf[i]
                 discounted_payback_period = i - 1 + fractional_year
            break

    return {
        "NPV": npv,
        "IRR": irr,
        "PP": payback_period,
        "DPP": discounted_payback_period
    }

# --- Hàm gọi API Gemini để trích xuất dữ liệu (Task 1) ---
def extract_financial_parameters(file_bytes, file_mime_type, api_key):
    """Trích xuất các tham số tài chính từ nội dung file Word bằng Gemini API (JSON output)."""
    
    if not api_key:
        raise ValueError("API Key không được cấu hình.")

    try:
        client = genai.Client(api_key=api_key)
        
        # System Instruction: Định rõ vai trò và yêu cầu format JSON
        system_instruction = (
            "Bạn là một chuyên gia tài chính. Nhiệm vụ của bạn là đọc file mô tả dự án "
            "và trích xuất các tham số quan trọng sau vào định dạng JSON. "
            "Dữ liệu phải là số. WACC và Thuế phải là giá trị thập phân (ví dụ: 10% là 0.1)."
        )

        # JSON Schema bắt buộc để đảm bảo đầu ra nhất quán
        response_schema = {
            "type": "OBJECT",
            "properties": {
                "project_name": {"type": "STRING", "description": "Tên hoặc mô tả ngắn gọn của dự án."},
                "investment_cost": {"type": "NUMBER", "description": "Vốn đầu tư ban đầu (CF0)."},
                "project_lifetime_years": {"type": "INTEGER", "description": "Dòng đời dự án theo năm."},
                "annual_revenue": {"type": "NUMBER", "description": "Doanh thu hàng năm (R), giả định không đổi."},
                "annual_cost_excluding_tax": {"type": "NUMBER", "description": "Chi phí hoạt động hàng năm (C), chưa bao gồm thuế."},
                "wacc_rate": {"type": "NUMBER", "description": "Chi phí vốn bình quân (WACC), định dạng thập phân (ví dụ: 0.1 cho 10%)."},
                "tax_rate": {"type": "NUMBER", "description": "Thuế suất thu nhập doanh nghiệp, định dạng thập phân (ví dụ: 0.2 cho 20%)."}
            },
            "required": [
                "project_name", "investment_cost", "project_lifetime_years", "annual_revenue", 
                "annual_cost_excluding_tax", "wacc_rate", "tax_rate"
            ]
        }
        
        prompt = (
            "Hãy trích xuất 7 thông tin tài chính quan trọng sau từ file Word đính kèm: "
            "1. Tên dự án, 2. Vốn đầu tư, 3. Dòng đời dự án (năm), 4. Doanh thu hàng năm, "
            "5. Chi phí hoạt động hàng năm, 6. Chi phí vốn (WACC), 7. Thuế suất. "
            "Đảm bảo kết quả đầu ra là JSON hợp lệ theo Schema đã cho."
        )

        # Multimodal request payload: Gửi file bytes và prompt cùng lúc
        contents = [
            {
                "inlineData": {
                    "data": file_bytes,
                    "mimeType": file_mime_type # Ví dụ: application/vnd.openxmlformats-officedocument.wordprocessingml.document
                }
            },
            {"text": prompt}
        ]
        
        config = {
            "responseMimeType": "application/json",
            "responseSchema": response_schema,
        }
        
        # Áp dụng Exponential Backoff để tăng độ tin cậy của API call
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=contents,
                    system_instruction=system_instruction,
                    config=config,
                )
                
                # Trả về kết quả nếu thành công
                return json.loads(response.text)
            except (APIError, json.JSONDecodeError) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    raise e

    except APIError as e:
        st.error(f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}")
        return None
    except json.JSONDecodeError:
        st.error("Lỗi giải mã JSON từ AI. Vui lòng đảm bảo file Word có cấu trúc rõ ràng và các thông số là số.")
        return None
    except Exception as e:
        st.error(f"Đã xảy ra lỗi không xác định trong quá trình trích xuất: {e}")
        return None

# --- Hàm gọi API Gemini để phân tích (Task 4) ---
def get_project_analysis(metrics_data, cash_flow_df, api_key):
    """Phân tích các chỉ số hiệu quả dự án bằng Gemini AI."""
    
    if not api_key:
        return "Lỗi: API Key chưa được cấu hình."

    try:
        client = genai.Client(api_key=api_key)
        
        # Chuyển đổi dữ liệu sang định dạng Markdown để AI dễ đọc
        metrics_markdown = pd.DataFrame(metrics_data.items(), columns=["Chỉ số", "Giá trị"]).to_markdown(index=False)
        cash_flow_markdown = cash_flow_df.to_markdown(index=False)

        prompt = f"""
        Bạn là một Chuyên gia Thẩm định Dự án Đầu tư. Dựa trên các chỉ số hiệu quả dự án và bảng dòng tiền sau, 
        hãy đưa ra một nhận xét phân tích chuyên sâu (khoảng 3-4 đoạn). 
        
        Phân tích cần tập trung vào:
        1. Tính khả thi của dự án (dựa trên NPV và IRR so với WACC).
        2. Rủi ro và tính thanh khoản (dựa trên PP và DPP).
        3. Khuyến nghị (Nên chấp nhận, từ chối hay cần xem xét thêm).
        
        --- Bảng Chỉ số Hiệu quả ---
        {metrics_markdown}
        
        --- Bảng Dòng tiền (Summary) ---
        {cash_flow_markdown}
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- Giao diện Streamlit ---

# 1. Tải File Word
st.subheader("1. Tải file Word chứa Phương án Kinh doanh (.docx)")
uploaded_file = st.file_uploader(
    "Tải file Word để AI trích xuất thông tin (Hỗ trợ .docx)",
    type=['docx']
)

# Khởi tạo session state để lưu trữ dữ liệu giữa các lần tương tác
if 'params' not in st.session_state:
    st.session_state['params'] = None
if 'metrics' not in st.session_state:
    st.session_state['metrics'] = None
if 'cash_flow_df' not in st.session_state:
    st.session_state['cash_flow_df'] = None

# Nút Trích xuất
if uploaded_file is not None:
    if st.button("🚀 Lọc Thông tin Dự án bằng AI (Bước 1)", key="extract_btn", type="primary"):
        if not API_KEY:
            st.error("Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY' trong Streamlit Secrets. Vui lòng cấu hình.")
        else:
            # Đọc file bytes
            file_bytes = uploaded_file.getvalue()
            
            with st.spinner('AI đang đọc và trích xuất dữ liệu từ file Word...'):
                extracted_params = extract_financial_parameters(
                    file_bytes, 
                    uploaded_file.type, 
                    API_KEY
                )
            
            if extracted_params:
                st.session_state['params'] = extracted_params
                st.session_state['metrics'] = None # Reset metrics khi có thông số mới
                st.session_state['ai_analysis'] = None
                st.success("✅ Trích xuất thông tin thành công! Chuyển sang Bước 2 & 3.")

# 2. Hiển thị Tham số và Xây dựng Dòng tiền (Task 2)
if st.session_state['params']:
    params = st.session_state['params']
    
    st.markdown("---")
    st.subheader("2. Thông số Dự án đã Lọc và Bảng Dòng tiền")
    
    st.info(f"Dự án đang phân tích: **{params.get('project_name', 'N/A')}**")
    
    # Hiển thị các thông số đã lọc
    col_v, col_n, col_wacc, col_thue = st.columns(4)
    with col_v:
        st.metric("Vốn đầu tư (CF0)", f"{params['investment_cost']:,.0f}")
    with col_n:
        st.metric("Dòng đời Dự án (Năm)", f"{params['project_lifetime_years']}")
    with col_wacc:
        st.metric("WACC (Chi phí vốn)", f"{params['wacc_rate']*100:.2f}%")
    with col_thue:
        st.metric("Thuế suất", f"{params['tax_rate']*100:.0f}%")

    col_rev, col_cost = st.columns(2)
    with col_rev:
        st.metric("Doanh thu hàng năm (R)", f"{params['annual_revenue']:,.0f}")
    with col_cost:
        st.metric("Chi phí hàng năm (C)", f"{params['annual_cost_excluding_tax']:,.0f}")
        
    try:
        # Lấy giá trị từ dict params
        N = params['project_lifetime_years']
        R = params['annual_revenue']
        C = params['annual_cost_excluding_tax']
        T = params['tax_rate']
        I = params['investment_cost']
        WACC = params['wacc_rate']

        # Dòng tiền ròng hàng năm (Net Cash Flow - NCF)
        # NCF = (Doanh thu - Chi phí) * (1 - Thuế)
        ncf_yearly = (R - C) * (1 - T)
        
        # Tạo dòng tiền
        years = list(range(N + 1))
        cash_flows_list = [-I] + [ncf_yearly] * N # CF[0] là Vốn đầu tư (số âm)
        
        # Tạo DataFrame cho bảng dòng tiền
        cash_flow_df = pd.DataFrame({
            "Năm": years,
            "Dòng tiền ròng (NCF)": cash_flows_list,
            "Hệ số Chiết khấu": [1/((1 + WACC)**i) for i in years],
        })
        
        # Tính toán Dòng tiền Chiết khấu (DCF)
        cash_flow_df["Dòng tiền Chiết khấu (DCF)"] = cash_flow_df.apply(
            lambda row: row['Dòng tiền ròng (NCF)'] * row["Hệ số Chiết khấu"], axis=1
        )
        
        st.dataframe(
            cash_flow_df.style.format({
                "Dòng tiền ròng (NCF)": "{:,.0f}",
                "Hệ số Chiết khấu": "{:.4f}",
                "Dòng tiền Chiết khấu (DCF)": "{:,.0f}",
            }), 
            use_container_width=True
        )
        st.session_state['cash_flow_df'] = cash_flow_df
        
    except KeyError:
        st.error("Lỗi: Thiếu tham số cần thiết để xây dựng dòng tiền. Vui lòng kiểm tra file Word.")
    except Exception as e:
        st.error(f"Lỗi khi xây dựng dòng tiền: {e}")
        
# 3. Tính toán và Hiển thị các Chỉ số Đánh giá (Task 3)
if st.session_state['cash_flow_df'] is not None:
    st.markdown("---")
    st.subheader("3. Các Chỉ số Đánh giá Hiệu quả Dự án")
    
    try:
        cash_flows = st.session_state['cash_flow_df']['Dòng tiền ròng (NCF)'].tolist()
        wacc = params['wacc_rate']
        
        # Tính toán các chỉ số 
        metrics = calculate_project_metrics(cash_flows, wacc)
        st.session_state['metrics'] = metrics
        
        # Hiển thị kết quả
        col_npv, col_irr, col_pp, col_dpp = st.columns(4)
        
        with col_npv:
            st.metric("NPV (Giá trị hiện tại ròng)", f"{metrics['NPV']:,.0f}")
        with col_irr:
            st.metric("IRR (Tỷ suất sinh lời nội bộ)", f"{metrics['IRR']*100:.2f}%")
        with col_pp:
            st.metric("PP (Thời gian hoàn vốn)", f"{metrics['PP']:.2f} năm")
        with col_dpp:
            st.metric("DPP (Hoàn vốn chiết khấu)", f"{metrics['DPP']:.2f} năm")
            
    except Exception as e:
        st.error(f"Lỗi khi tính toán các chỉ số: {e}")

# 4. Phân tích AI (Task 4)
if st.session_state['metrics'] is not None:
    st.markdown("---")
    st.subheader("4. Phân tích Chỉ số Hiệu quả Dự án (AI)")
    
    if st.button("🧠 Yêu cầu AI Phân tích Hiệu quả (Bước 2)", key="analyze_btn"):
        if not API_KEY:
            st.error("Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng cấu hình.")
        else:
            with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                # Chuẩn bị dữ liệu cho AI
                metrics_data = {
                    "NPV": f"{st.session_state['metrics']['NPV']:,.0f}",
                    "IRR": f"{st.session_state['metrics']['IRR']*100:.2f}%",
                    "WACC": f"{params['wacc_rate']*100:.2f}%",
                    "PP": f"{st.session_state['metrics']['PP']:.2f} năm",
                    "DPP": f"{st.session_state['metrics']['DPP']:.2f} năm",
                    "Dòng đời Dự án": f"{params['project_lifetime_years']} năm"
                }
                
                # Gửi 4 dòng đầu và dòng cuối của Cash Flow để AI có cái nhìn tổng quan
                df_to_analyze = st.session_state['cash_flow_df'].iloc[0:4]
                if len(st.session_state['cash_flow_df']) > 4:
                    df_to_analyze = pd.concat([df_to_analyze, st.session_state['cash_flow_df'].tail(1)], ignore_index=True)
                
                analysis_result = get_project_analysis(
                    metrics_data, 
                    df_to_analyze,
                    API_KEY
                )
                
                st.session_state['ai_analysis'] = analysis_result

    if 'ai_analysis' in st.session_state:
        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
        st.info(st.session_state['ai_analysis'])

st.markdown("---")
st.caption("⚠️ Lưu ý: Ứng dụng này hoạt động dựa trên giả định **Dòng tiền ròng hàng năm không đổi**. Để sử dụng, bạn cần có Khóa API Gemini được cấu hình trong Streamlit Secrets với tên là `GEMINI_API_KEY`.")
