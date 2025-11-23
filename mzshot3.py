import cv2
import mediapipe as mp
import av
import time
import queue
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# ---------------- 1. 기본 설정 ----------------
st.set_page_config(page_title="AI 자동 캡처", layout="centered")
st.title("✌️ 제스처 자동 캡처 카메라")
st.write("카메라를 켜고 **'V' 제스처**를 하세요. 자동으로 찍히고 다운로드 버튼이 뜹니다.")

# STUN 서버 (이게 없으면 배포 시 검은 화면만 뜸)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Mediapipe 로드
mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ---------------- 2. 영상 처리 로직 (백그라운드 실행) ----------------
class VictoryProcessor(VideoTransformerBase):
    def __init__(self):
        self.face_detector = mp_face.FaceDetection(min_detection_confidence=0.6)
        self.hand_detector = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6)
        self.result_queue = queue.Queue() # 메인 화면으로 사진을 보내는 통로
        self.last_capture_time = 0
        self.cooldown = 3.0  # 3초 쿨타임

    def is_victory(self, lms, w, h):
        # 좌표 변환 함수
        def c(i):
            lm = lms.landmark[i]
            return int(lm.x * w), int(lm.y * h)

        try:
            # 검지(8), 중지(12) 끝
            i_tip, m_tip = c(8), c(12)
            # 약지(16), 새끼(20) 끝
            r_tip, p_tip = c(16), c(20)
            # 각 손가락 마디
            i_kn, m_kn = c(5), c(9)
            r_kn, p_kn = c(13), c(17)

            # 검지/중지는 펴지고(위), 약지/새끼는 접힘(아래)
            # (화면상 위쪽일수록 y값이 작음)
            if (i_tip[1] < i_kn[1] and m_tip[1] < m_kn[1] and 
                r_tip[1] > r_kn[1] and p_tip[1] > p_kn[1]):
                return True
        except:
            pass
        return False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 이미지 복사 (쓰기 가능하게)
        img_out = img.copy()
        img_rgb = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = img_out.shape

        # 1. 얼굴 감지
        face_res = self.face_detector.process(img_rgb)
        face_detected = face_res.detections is not None

        # 2. 손 감지
        hand_res = self.hand_detector.process(img_rgb)
        victory_detected = False

        if hand_res.multi_hand_landmarks:
            for handLms in hand_res.multi_hand_landmarks:
                mp_draw.draw_landmarks(img_out, handLms, mp_hands.HAND_CONNECTIONS)
                if self.is_victory(handLms, img_w, img_h):
                    victory_detected = True

        # 3. 캡처 조건: 얼굴 O + 브이 O + 쿨타임 지남
        current_time = time.time()
        if face_detected and victory_detected:
            if current_time - self.last_capture_time > self.cooldown:
                self.last_capture_time = current_time
                
                # 화면에 텍스트 표시
                cv2.putText(img_out, "CAPTURED!", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
                
                # 중요: 캡처된 원본(깨끗한 이미지)를 큐에 넣음
                # 선이 그려지지 않은 원본을 저장하고 싶으면 'img'를 사용, 
                # 선 그려진 걸 원하면 'img_out'을 사용. 여기선 'img_out' 사용.
                self.result_queue.put(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))

        return av.VideoFrame.from_ndarray(img_out, format="bgr24")

# ---------------- 3. 스트리밍 및 UI 실행 ----------------
ctx = webrtc_streamer(
    key="snapshot-camera",
    video_processor_factory=VictoryProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False}
)

# 실시간으로 큐 확인하여 사진이 찍혔는지 감시
if ctx.video_processor:
    if not ctx.video_processor.result_queue.empty():
        # 큐에서 사진 꺼내기
        captured_img = ctx.video_processor.result_queue.get()
        
        # 화면에 표시
        st.success("📸 찰칵! 사진이 촬영되었습니다.")
        st.image(captured_img, caption="캡처된 이미지", use_column_width=True)
        
        # 다운로드 버튼 생성 (이미지 -> 바이트 변환)
        try:
            is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(captured_img, cv2.COLOR_RGB2BGR))
            if is_success:
                st.download_button(
                    label="⬇️ 사진 저장하기 (Click to Save)",
                    data=buffer.tobytes(),
                    file_name=f"capture_{int(time.time())}.jpg",
                    mime="image/jpeg"
                )
        except Exception as e:
            st.error(f"저장 준비 중 오류: {e}")

st.markdown("---")
st.caption("PC: Chrome 권장 | Mobile: Safari/Chrome 권장 | 얼굴과 손이 모두 나와야 찍힙니다.")