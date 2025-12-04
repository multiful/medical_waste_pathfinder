# HIRA – 폐렴 환자 진료 현황 대시보드

건강보험심사평가원(HIRA) 공개 자료를 활용해 **폐렴 입원 환자의 진료·결과 지표를 시각화**한 Streamlit 대시보드입니다.  
의료기관/연령대/기간별 지표를 한 화면에서 비교하면서, 진료 품질과 자원 사용 패턴을 쉽게 파악하는 것이 목표입니다.

---

## 대시보드 미리보기

<img width="1854" height="861" alt="image" src="https://github.com/user-attachments/assets/32677f5d-c74e-4091-805c-c47c096e5c9f" />

---

## Link

- **Demo URL**  
  [https://visualproject-w2fzqfdpintkrnpyasxpzk.streamlit.app/](https://visualproject-w2fzqfdpintkrnpyasxpzk.streamlit.app/)

- **발표자료(PDF)**  
  [발표자료 PDF](https://github.com/user-attachments/files/23655893/3805_2_3_8_NBD.pdf)

- **FIGMA 프로토타입**  
  [UI/UX 설계 Figma 보기](https://www.figma.com/proto/dS9CrD2hsLZzkvkAgFc4xi?node-id=0-1&t=BGqAVemjihn0riOO-6)

---

## 프로젝트 개요

- **목적**  
  - 폐렴 입원 환자의 진료량·결과 지표를 시각화하여, 병원·연령·기간별 특성을 직관적으로 파악
  - 의료품질 평가 및 정책/경영 의사결정 시 참고할 수 있는 분석 도구 제공

- **대상 사용자**  
  - 보건의료 데이터 분석을 수행하는 연구자 / 학생  
  - 의료기관 및 정책 담당자(가상의 타깃)

---

## 주요 기능

- 기간, 연령대, 성별, 기관 유형 등 **필터링 기능**
- 입원 환자 수, 평균 재원일수, 사망률 등 **핵심 지표 카드** 제공
- 기관/연령대별 **막대 그래프·라인 차트**로 트렌드 비교
- 선택한 조건에 대해 **상세 테이블** 및 다운로드(csv) 기능(구현했다면 기재)
- Figma 기반으로 설계한 레이아웃을 Streamlit에 반영한 UI

---

## 데이터

- **데이터 출처**: HIRA(건강보험심사평가원) 공개 자료 및 강의/과제에서 제공된 가공 데이터
- **주요 컬럼 예시**  
  - 기관 ID/유형, 연도·분기/월  
  - 환자 수, 입원 건수, 평균 재원일수  
  - 사망률/재입원률 등 결과 지표(해당되는 것만 적기)

(필요하면 실제 쓰는 컬럼 이름으로 수정해서 적으면 좋아.)

---

## 기술 스택

- **Dashboard**: Streamlit
- **Language**: Python
- **Visualization**: Plotly 
- **Design**: Figma
- **Version Control**: Git / GitHub

---

## 내 역할

- 폐렴 환자 대시보드 **지표 선정 및 화면 레이아웃 설계**
- Figma를 활용한 **UI 프로토타입 제작** 및 피드백 반영
- Streamlit 기반 **대시보드 구현**
  - 필터(기간/연령/기관) 로직 및 쿼리 구성
  - 주요 카드 지표·그래프·테이블 컴포넌트 구현
- 데이터 전처리 및 집계 코드 작성(pandas)
- 피쳐 엔지니어링 및 주요 피쳐 통계 분석

--- 

## 로컬 실행 방법

```bash
git clone https://github.com/multiful/HIRA_Streamlit.git
cd HIRA_Streamlit
pip install -r requirements.txt
streamlit run streamlit.py
