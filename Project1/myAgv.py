import cv2
import numpy as np
import time
from pymycobot import MyAgv

# agv.move_control( 128 기준으로 전진 후진, 좌/우 이동, 좌/우 회전)

# MyAgv 객체를 생성
agv = MyAgv("/dev/ttyAMA2", 115200)


def main():

    cap = cv2.VideoCapture(0)    # 웹캠에서 영상을 읽어오는 객체를 생성
    

    while cap.isOpened():
        ret, frame = cap.read() # 웹캠에서 영상을 읽어옴

        if not ret:
            print("Camera Error")
            break

        crop_img = frame[200:480,]  # 차선 검출을 위해 영상을 자름 (가로: 200, 세로: 480)

        cv2.imshow("normal", crop_img)  # 자른 영상을 출력

        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV) # HSV 색 공간으로 변환

        lower_yellow = np.array([20, 100, 100]) # 노런색의 하한갑을 지정
        upper_yellow = np.array([40, 255, 255]) # 노란색의 상한 값을 지정
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow) # 노란색에 해당하는 영역을 마스킹

        # cv2.findContours() 함수를 사용하여 마스크에서 노란색에 해당하는 컨투어를 찾는다
        contours, hierarchy = cv2.findContours(mask.copy(), 1, cv2.CHAIN_APPROX_NONE)

        cmt = 0         # 중심점 x좌표 초기값 
        CL_count = 0    # 왼쪽으로 이동한 횟수  
        CR_count = 0    # 오른쪽으로 이동한 횟수

        if len(contours) > 0:   # 노란색에 해당하는 컨투어가 존재할 때
            biggest_contour = max(contours, key = cv2.contourArea)    # 가장 큰 컨투어를 선택
            M = cv2.moments(biggest_contour)             # 컨투어의 모멘트를 계산 

            if M['m00'] != 0:      # 모멘트의 중심이 0이 아닐 때
                cx = int(M['m10'] / M['m00'])   # 중심점 x좌표를 계산함
                cmt = cx    # 중심점 x좌표를 업데이트 함

                print(cx)
                CL_count = 0
                CR_count = 0

                if cx <= 220:   # 중심점이 왼쪽에 있을 때
                    print("Turn Left")
                    cv2.putText(mask, "Left", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if cx <= 190:   # 더 완쪽에 있을 때
                        agv.move_control(128, 128, 135) # 좌화전 모터 명령을 전송
                        time.sleep(0.1) # 0.1초 대기

                    else:
                        agv.move_control(128, 128, 130) # 좌회전 모터 명령을 전송
                        time.sleep(0.05)    # 0.05초 대기

                    cv2.putText(mask, "Go", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    agv.move_control(129, 218, 128) # 직진 모터 명령을 전송
                    time.sleep(0.05)    # 0.05초 대기

                elif cx >= 420:     # 중심정이 오른쪽에 있을 때
                    print("Turn Right")
                    cv2.putText(mask, "Right", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if cx >= 450:   # 더 오른쪽에 있을 때
                        agv.move_control(128, 128, 120) # 우회전 모터 명령을 전송
                        time.sleep(0.1) # 0.1초 대기
                    else:
                        agv.move_control(128, 128, 126) # 우회전 모터 명령을 전송
                        time.sleep(0.05)    # 0.05초 대기

                    agv.move_control(129, 128, 128) # 직진 모터 명령을 전송
                    time.sleep(0.05)    # 0.05초 대기

                else:   # 중심점이 가운데에 있을 때
                    print("GO")
                    cv2.putText(mask, "Go", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    agv.move_control(129, 128, 128) # 직진 모터 명령을 전송
                    time.sleep(0.05)    # 0.05초 대기
            
        else:   # 노란색 컨투어가 존재하지 않을 때
            if cmt <= 320:  # 중심점이 왼쪽에 있을 때
                if CL_count == 10:  # 왼쪽으로 이동한 횟수가 10일때
                    print("Stop Turn Left")
                    agv.stop()  # 모터를 정지
                    
                else:   # 왼쪽으로 이동한 횟수가 10이 아닐 때
                    CL_count += 1   #  왼쪽으로 이동한 횟수를 증가 시킴
                    agv.move_control(128, 128, 131) # 죄회전 모터 명령을 전송
                    time.sleep(0.05)    # 0.05초 대기

            elif cmt > 320: # 중심정이 오른쪽에 있을 때
                if CR_count == 10:  # 오른쪽으로 이동한 횟수가 10일 때
                    print("Stop Turn Right")
                    agv.stop()  # 모터를 정지

                else:   #오른족으로 이동한 횟수가 10이 아닐 때
                    CR_count += 1   # 오른쪽으로 이동한 횟수를 증가 시킴
                    agv.movecontrol(128, 128, 125)  # 우회전 모터 명령을 전송
                    time.sleep(0.05)    # 0.05초 대기

        if cv2.waitKey(1) & 0xFF == ord('q'):
            agv.stop()
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()