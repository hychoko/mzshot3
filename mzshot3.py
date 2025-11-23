import cv2
import mediapipe as mp
import av
import time
import queue
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# ---------------- 1. 페이지 설정 및 세션 초기화 ----------------
st.set_page_config(page_title="자동 제스처 카메라", layout="centered")

# 찍은 사진을 저장할 공간 (새로고침 되어도 유지됨)
if "captured_image" not in st.session_state:
    st.session_state["captured_image"] = None

st.title("✌️ 제스처 자동 캡처")
st.write("카메라를 켜고 **V 사인**을 해보세요. 자동으로 찍히고 사진이 뜹니다.")

# STUN 서버 (검은 화면 방지)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Mediapipe 초기화
mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ---------------- 2. 백그라운드 영상 처리기 ----------------
class VictoryProcessor(VideoTransformerBase):
    def __init__(self):
        self.face_detector = mp_face.FaceDetection(min_detection_confidence=0.6)
        self.hand_detector = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6)
        self.result_queue = queue.Queue() # 메인 화면으로 사진 보내는 통로
        self.last_capture_time = 0
        self.captured = False # 중복 촬영 방지 플래그

    def is_victory(self, lms, w, h):
        def c(i):
            lm = lms.landmark[i]
            return int(lm.x * w), int(lm.y * h)
        try:
            # 검지(8), 중지(12) 펴짐 (Y좌표가 낮음) / 약지, 새끼 접힘 (Y좌표가 높음)
            if (lms.landmark[8].y < lms.landmark[5].y and 
                lms.landmark[12].y < lms.landmark[9].y and 
                lms.landmark[16].y > lms.landmark[13].y and 
                lms.landmark[20].y > lms.landmark[17].y):
                return True
        except:
            pass
        return False

    def recv(self, frame):
        # 이미 캡처했으면 처리 중단 (UI 업데이트 대기)
        if self.captured:
            return av.VideoFrame.from_ndarray(frame.to_ndarray(format="bgr24"), format="bgr24")

        img = frame.to_ndarray(format="bgr24")
        img_out = img.copy()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        # 얼굴 & 손 인식
        face_res = self.face_detector.process(img_rgb)
        hand_res = self.hand_detector.process(img_rgb)
        
        face_detected = face_res.detections is not None
        victory_detected = False

        if hand_res.multi_hand_landmarks:
            for handLms in hand_res.multi_hand_landmarks:
                mp_draw.draw_landmarks(img_out, handLms, mp_hands.HAND_CONNECTIONS)
                if self.is_victory(handLms, w, h):
                    victory_detected = True

        # ★ 조건 만족 시 자동 캡처 ★
        if face_detected and victory_detected:
            current_time = time.time()
            if current_time - self.last_capture_time > 2.0: # 쿨타임 2초
                self.last_capture_time = current_time
                self.captured = True # 플래그 세움
                
                # 캡처 효과 텍스트
                cv2.putText(img_out, "CAPTURED!", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
                
                # 큐(Queue)에 원본 사진 전송
                self.result_queue.put(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        return av.VideoFrame.from_ndarray(img_out, format="bgr24")

# ---------------- 3. 메인 화면 UI 로직 ----------------

# WebRTC 스트리머 실행
ctx = webrtc_streamer(
    key="gesture-cam",
    video_processor_factory=VictoryProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False}
)

# ★ 자동 감지 루프 ★
# 카메라가 켜져 있는 동안, 백그라운드에서 사진이 넘어오는지 계속 감시합니다.
if ctx.state.playing:
    placeholder = st.empty() # 상태 메시지 표시용
    while True:
        if ctx.video_processor:
            try:
                # 큐에서 사진이 들어왔는지 확인 (대기 시간 0.1초)
                result_img = ctx.video_processor.result_queue.get(timeout=0.1)
            except queue.Empty:
                result_img = None
            
            # 사진이 도착했다면?
            if result_img is not None:
                # 세션에 저장하고 화면 새로고침 (Rerun)
                st.session_state["captured_image"] = result_img
                ctx.video_processor.captured = False # 캡처 플래그 초기화
                st.rerun() # ★ 여기서 자동으로 화면이 갱신됩니다!
                break
        
        time.sleep(0.05) # CPU 과부하 방지

# ---------------- 4. 캡처된 사진 및 저장 버튼 표시 ----------------
# 화면이 새로고침되면 실행되는 부분
st.markdown("---")
if st.session_state["captured_image"] is not None:
    st.success("📸 캡처 완료! 아래 버튼을 눌러 저장하세요.")
    
    # 1. 사진 보여주기
    st.image(st.session_state["captured_image"], caption="방금 찍은 사진", use_column_width=True)
    
    # 2. 다운로드 버튼 생성 (이미지 -> 파일 변환)
    img_bgr = cv2.cvtColor(st.session_state["captured_image"], cv2.COLOR_RGB2BGR)
    ret, buffer = cv2.imencode(".jpg", img_bgr)
    if ret:
        st.download_button(
            label="⬇️ 사진 저장하기 (Click to Save)",
            data=buffer.tobytes(),
            file_name=f"capture_{int(time.time())}.jpg",
            mime="image/jpeg"
        )
