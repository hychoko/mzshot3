import cv2
import mediapipe as mp
import av
import time
import queue
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# ---------------- 1. 기본 설정 및 세션 초기화 ----------------
st.set_page_config(page_title="AI 자동 캡처", layout="centered")

# 사진 저장소 초기화 (새로고침 해도 사진 유지되도록)
if "captured_image" not in st.session_state:
    st.session_state["captured_image"] = None

st.title("✌️ 제스처 자동 캡처 카메라")
st.write("카메라를 켜고 **'V' 제스처**를 하세요. 3초 뒤 다시 촬영 가능합니다.")

# STUN 서버 설정
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Mediapipe 로드
mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ---------------- 2. 영상 처리 로직 (백그라운드) ----------------
class VictoryProcessor(VideoTransformerBase):
    def __init__(self):
        self.face_detector = mp_face.FaceDetection(min_detection_confidence=0.6)
        self.hand_detector = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6)
        self.result_queue = queue.Queue() # 메인 화면으로 사진을 보내는 우체통
        self.last_capture_time = 0
        self.cooldown = 3.0  # 쿨타임

    def is_victory(self, lms, w, h):
        def c(i):
            lm = lms.landmark[i]
            return int(lm.x * w), int(lm.y * h)
        try:
            # 검지(8), 중지(12) 펴짐 / 약지(16), 새끼(20) 접힘
            i_tip, m_tip = c(8), c(12)
            r_tip, p_tip = c(16), c(20)
            i_kn, m_kn = c(5), c(9)
            r_kn, p_kn = c(13), c(17)
            
            return (i_tip[1] < i_kn[1] and m_tip[1] < m_kn[1] and 
                    r_tip[1] > r_kn[1] and p_tip[1] > p_kn[1])
        except:
            return False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_out = img.copy()
        img_rgb = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = img_out.shape

        # 얼굴 & 손 인식
        face_res = self.face_detector.process(img_rgb)
        hand_res = self.hand_detector.process(img_rgb)
        
        face_detected = face_res.detections is not None
        victory_detected = False

        if hand_res.multi_hand_landmarks:
            for handLms in hand_res.multi_hand_landmarks:
                mp_draw.draw_landmarks(img_out, handLms, mp_hands.HAND_CONNECTIONS)
                if self.is_victory(handLms, img_w, img_h):
                    victory_detected = True

        # 캡처 로직
        current_time = time.time()
        if face_detected and victory_detected:
            if current_time - self.last_capture_time > self.cooldown:
                self.last_capture_time = current_time
                
                # 'CAPTURED' 텍스트 그리기
                cv2.putText(img_out, "CAPTURED!", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
                
                # ★ 중요: 큐(우체통)에 사진 넣기
                # img: 선 없는 원본 / img_out: 선 그려진 버전
                self.result_queue.put(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
                
        return av.VideoFrame.from_ndarray(img_out, format="bgr24")

# ---------------- 3. 메인 화면 UI 및 대기 로직 ----------------

# 웹캠 스트리머 실행
ctx = webrtc_streamer(
    key="snapshot-camera",
    video_processor_factory=VictoryProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False}
)

# ★ 여기가 핵심: 카메라가 켜져 있는 동안 사진이 오나 계속 감시하는 루프
if ctx.state.playing:
    # 큐에서 데이터를 꺼내올 때까지 반복
    while True:
        if ctx.video_processor:
            try:
                # 0.1초 동안 기다려봄
                result = ctx.video_processor.result_queue.get(timeout=0.1)
            except queue.Empty:
                result = None
            
            # 사진이 도착했으면?
            if result is not None:
                st.session_state["captured_image"] = result # 저장
                st.rerun() # 화면 새로고침 (즉시 표시하기 위해)
                break # 루프 탈출
        
        # CPU 과부하 방지용 잠시 대기
        time.sleep(0.1) 

# ---------------- 4. 사진 및 다운로드 버튼 표시 ----------------
st.markdown("---")
if st.session_state["captured_image"] is not None:
    st.success("📸 사진이 촬영되었습니다!")
    
    # 이미지 표시
    st.image(st.session_state["captured_image"], caption="방금 찍은 사진", use_column_width=True)
    
    # 다운로드 버튼 만들기
    # 이미지를 바이트로 변환
    img_bgr = cv2.cvtColor(st.session_state["captured_image"], cv2.COLOR_RGB2BGR)
    is_success, buffer = cv2.imencode(".jpg", img_bgr)
    
    if is_success:
        st.download_button(
            label="⬇️ 사진 내 컴퓨터에 저장하기",
            data=buffer.tobytes(),
            file_name=f"capture_{int(time.time())}.jpg",
            mime="image/jpeg"
        )
else:
    st.write("아직 찍힌 사진이 없습니다. V를 해보세요!")
