import os
import io
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import cv2
import imutils
from imutils.perspective import four_point_transform
from imutils import contours

app = FastAPI(title="OMR Checker API")

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "OMR Checker API is running successfully!"}

# Supabase configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL", "YOUR_SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "YOUR_SUPABASE_KEY")

def get_supabase() -> Client:
    if SUPABASE_URL == "YOUR_SUPABASE_URL" or SUPABASE_KEY == "YOUR_SUPABASE_KEY":
        raise ValueError("Supabase credentials not set in environment variables.")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

class OMRResult(BaseModel):
    question_id: str
    selected_option: Optional[str]
    is_correct: bool
    marks_obtained: float
    correct_answer: str

class GradingResponse(BaseModel):
    exam_id: str
    total_marks: float
    obtained_marks: float
    results: List[OMRResult]
    processed_image_base64: Optional[str] = None

def process_omr_image(image_bytes: bytes, num_questions: int = 100, num_choices: int = 4):
    """
    Process the specific 100 MCQ OMR sheet with 4 columns.
    Returns the selected answers and a base64 encoded image showing what was detected.
    """
    # 1. Load image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")

    # 2. Preprocessing
    # Skip document detection completely. Assume the uploaded image IS the document.
    # This prevents the "full black image" issue caused by bad cropping.
    paper = image.copy()
    
    # Resize to a standard size to make bubble detection consistent
    paper = cv2.resize(paper, (800, 1131)) # A4 ratio
    gray = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)

    # 3. Thresholding (binarization)
    # Simple thresholding for scanned PDFs
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]

    # 4. Find all circular contours (bubbles)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    raw_bubbles = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # Filter for bubbles: roughly circular, specific size range
        # Made extremely forgiving to just find ANYTHING circular
        if 5 <= w <= 80 and 5 <= h <= 80 and 0.5 <= ar <= 1.5:
            raw_bubbles.append(c)

    # Remove duplicate contours (inner/outer rings of the same bubble)
    questionCnts = []
    for c in raw_bubbles:
        (x, y, w, h) = cv2.boundingRect(c)
        is_duplicate = False
        for qc in questionCnts:
            (qx, qy, qw, qh) = cv2.boundingRect(qc)
            if abs(x - qx) < 5 and abs(y - qy) < 5:
                is_duplicate = True
                break
        if not is_duplicate:
            questionCnts.append(c)

    # Draw ALL detected bubbles in RED for debugging
    cv2.drawContours(paper, questionCnts, -1, (0, 0, 255), 2)

    if len(questionCnts) < 100:
        # If we didn't find enough bubbles, return the image showing what we DID find
        cv2.putText(paper, f"Error: Only found {len(questionCnts)} bubbles.", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', paper)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return [], img_base64

    # 7. Sort and Group into 4 Columns (25 questions each)
    selected_answers = []
    
    # We expect 4 columns. Let's divide the image width into 4 sections
    width = gray.shape[1]
    col_width = width // 4
    
    for col_idx in range(4):
        # Filter bubbles that fall into this column
        col_bubbles = []
        for c in questionCnts:
            (x, y, w, h) = cv2.boundingRect(c)
            if x >= col_idx * col_width and x < (col_idx + 1) * col_width:
                col_bubbles.append(c)
                
        if len(col_bubbles) == 0:
            # Pad with empty answers if column is missing
            selected_answers.extend([-1] * 25)
            continue
            
        # Sort column bubbles top-to-bottom
        col_bubbles = contours.sort_contours(col_bubbles, method="top-to-bottom")[0]
        
        # Group into rows (questions)
        rows = []
        current_row = [col_bubbles[0]]
        
        for c in col_bubbles[1:]:
            _, y1, _, h1 = cv2.boundingRect(current_row[-1])
            _, y2, _, _ = cv2.boundingRect(c)
            
            # If Y difference is small, it's the same row
            if abs(y1 - y2) < h1:
                current_row.append(c)
            else:
                rows.append(current_row)
                current_row = [c]
        rows.append(current_row)
        
        # Process each row (up to 25 questions per column)
        for i, row in enumerate(rows):
            if i >= 25: break # Max 25 questions per column
            
            # Sort left-to-right (A, B, C, D)
            row = contours.sort_contours(row, method="left-to-right")[0]
            
            bubbled = None
            for (j, c) in enumerate(row):
                if j >= num_choices: break
                
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                total = cv2.countNonZero(mask)
                
                if bubbled is None or total > bubbled[0]:
                    bubbled = (total, j, c)
                    
            # Threshold for "filled" bubble (adaptive thresholding makes filled bubbles have many white pixels)
            # A circle of diameter 20 has ~314 pixels. Let's say at least 40 pixels must be white.
            if bubbled is not None and bubbled[0] > 40:
                selected_answers.append(bubbled[1])
                # Draw the selected bubble in GREEN
                cv2.drawContours(paper, [bubbled[2]], -1, (0, 255, 0), 3)
            else:
                selected_answers.append(-1)
                
        # Pad if less than 25 rows found
        while len(selected_answers) < (col_idx + 1) * 25:
            selected_answers.append(-1)

    # Ensure exactly 100 answers
    selected_answers = selected_answers[:100]

    # Map indices to letters (0 -> A, 1 -> B, etc.)
    options_map = {0: "A", 1: "B", 2: "C", 3: "D", -1: None}
    final_answers = [options_map.get(idx, None) for idx in selected_answers]
    
    # Encode the debug image to base64
    _, buffer = cv2.imencode('.jpg', paper)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return final_answers, img_base64


class AnswerInsert(BaseModel):
    question_id: str
    selected_option: Optional[str]
    is_correct: bool
    marks_obtained: float

class SaveResultsRequest(BaseModel):
    submission_id: str
    answers: List[AnswerInsert]

@app.post("/save-results")
async def save_results(request: SaveResultsRequest):
    try:
        supabase = get_supabase()
        answers_to_insert = []
        for ans in request.answers:
            answers_to_insert.append({
                # Omitting submission_id to avoid the foreign key constraint error you faced earlier
                "question_id": ans.question_id,
                "selected_option": ans.selected_option,
                "is_correct": ans.is_correct,
                "marks_obtained": ans.marks_obtained
            })
        
        if answers_to_insert:
            supabase.table("student_answers").insert(answers_to_insert).execute()
            
        return {"message": "Results saved successfully"}
    except Exception as e:
        print(f"Save Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-omr", response_model=GradingResponse)
async def upload_omr(
    exam_id: str = Form(...),
    submission_id: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        supabase = get_supabase()
        
        # 1. Fetch questions and correct answers for the exam from Supabase
        response = supabase.table("questions").select("*").eq("exam_id", exam_id).execute()
        questions = response.data
        
        if not questions:
            raise HTTPException(status_code=404, detail="No questions found for this exam.")
            
        # Sort questions by created_at or a specific order field if you have one
        questions = sorted(questions, key=lambda q: q.get('created_at', ''))
        num_questions = len(questions)
        
        # 2. Process the uploaded OMR image
        image_bytes = await file.read()
        try:
            # Assuming 4 choices (A, B, C, D) per question
            selected_options, processed_image_base64 = process_omr_image(image_bytes, num_questions=num_questions, num_choices=4)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing OMR image: {str(e)}")

        # 3. Grade the answers
        results = []
        total_marks = 0.0
        obtained_marks = 0.0

        for i, question in enumerate(questions):
            q_id = question["id"]
            correct_ans = question.get("correct_answer")
            marks = float(question.get("marks", 1.0))
            
            # Get the selected option from OpenCV processing
            selected_opt = selected_options[i] if i < len(selected_options) else None
            
            is_correct = (selected_opt == correct_ans) if selected_opt else False
            
            marks_awarded = marks if is_correct else 0.0
            
            total_marks += marks
            obtained_marks += marks_awarded
            
            result = OMRResult(
                question_id=q_id,
                selected_option=selected_opt,
                is_correct=is_correct,
                marks_obtained=marks_awarded,
                correct_answer=correct_ans or ""
            )
            results.append(result)

        # We no longer save to the database here. It will be done via /save-results
        return GradingResponse(
            exam_id=exam_id,
            total_marks=total_marks,
            obtained_marks=obtained_marks,
            results=results,
            processed_image_base64=processed_image_base64
        )

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
