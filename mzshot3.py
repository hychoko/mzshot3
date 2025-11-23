import cv2
import mediapipe as mp
import av
import time
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# ---------------- 1. 기본 설정 ----------------
st.set_page_config(page_title="제스처 캡처", layout="centered")
st.title("✌️ 제스처 캡처 카메라")

# 세션 상태 초기화 (찍은 사진 저장용)
if "snapshot" not in st.session_state:
    st.session_state.snapshot = None

# STUN 서버 (배포 필수 설정)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ---------------- 2. 영상 처리 로직 ----------------
class VictoryProcessor(VideoTransformerBase):
    def __init__(self):
        self.mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.6)
        self.mp_hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.6)
        self.mp_draw = mp.solutions.drawing_utils
        self.last_capture_time = 0
        self.captured_frame = None  # 가장 최근 찍힌 프레임 저장

    def is_victory(self, lms, w, h):
        # 좌표 변환
        def c(i):
            lm = lms.landmark[i]
            return int(lm.x * w), int(lm.y * h)
        try:
            # 검지(8), 중지(12) 펴짐 / 약지(16), 새끼(20) 접힘 확인
            if (lms.landmark[8].y < lms.landmark[5].y and 
                lms.landmark[12].y < lms.landmark[9].y and 
                lms.landmark[16].y > lms.landmark[13].y and 
                lms.landmark[20].y > lms.landmark[17].y):
                return True
        except:
            pass
        return False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_out = img.copy() # 화면 출력용 (선 그리기)
        img_h, img_w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 얼굴/손 인식
        face_res = self.mp_face.process(img_rgb)
        hand_res = self.mp_hands.process(img_rgb)
        
        face_detected = face_res.detections is not None
        victory_detected = False

        if hand_res.multi_hand_landmarks:
            for handLms in hand_res.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img_out, handLms, mp.solutions.hands.HAND_CONNECTIONS)
                if self.is_victory(handLms, img_w, img_h):
                    victory_detected = True

        # 캡처 조건 충족 시
        current_time = time.time()
        if face_detected and victory_detected:
            # 3초 쿨타임
            if current_time - self.last_capture_time > 3.0:
                self.last_capture_time = current_time
                self.captured_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 원본(깨끗한 이미지) 저장

            # 캡처 직후 1초 동안 화면에 텍스트 표시
            if current_time - self.last_capture_time < 1.0:
                 cv2.putText(img_out, "CAPTURED!", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)

        return av.VideoFrame.from_ndarray(img_out, format="bgr24")

# ---------------- 3. 메인 UI ----------------
col1, col2 = st.columns([2, 1])

with col1:
    st.write("왼쪽 카메라에서 V를 하세요. 'CAPTURED'가 뜨면 오른쪽 버튼을 누르세요.")
    # 스트리머 실행 (key값 고정)
    ctx = webrtc_streamer(
        key="snapshot",
        video_processor_factory=VictoryProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False}
    )

with col2:
    st.write("### 📸 사진 확인")
    
    # [사진 가져오기] 버튼을 눌러야 프로세서 내부의 이미지를 가져옴
    if st.button("찍은 사진 가져오기"):
        if ctx.video_processor:
            if ctx.video_processor.captured_frame is not None:
                st.session_state.snapshot = ctx.video_processor.captured_frame
                st.success("사진을 가져왔습니다!")
            else:
                st.warning("아직 찍힌 사진이 없습니다. V 포즈를 취해보세요.")
    
    # 가져온 사진이 있으면 표시 및 다운로드 버튼 제공
    if st.session_state.snapshot is not None:
        st.image(st.session_state.snapshot, caption="결과물", use_column_width=True)
        
        # 이미지 -> 바이트 변환
        img_bgr = cv2.cvtColor(st.session_state.snapshot, cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode('.jpg', img_bgr)
        
        if ret:
            st.download_button(
                label="⬇️ 파일로 저장하기",
                data=buffer.tobytes(),
                file_name=f"selfie_{int(time.time())}.jpg",
                mime="image/jpeg"
            )

st.markdown("---")
st.caption("사용법: 1. 카메라 켜기 -> 2. V 포즈 (CAPTURED 뜸) -> 3. '찍은 사진 가져오기' 버튼 클릭 -> 4. 저장")
