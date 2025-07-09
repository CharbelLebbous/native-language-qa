import os
from loader import load_documents
from embedder import EmbeddingIndexer
from qa import answer_question

test_cases = [
    {
        "question": "What is Charbel's nationality?",
        "expected_keywords": ["lebanese"],
        "expected_file": "charbel.txt",
        "expected_page": "N/A",
    },     
    {
        "question": "What is the status of Charbel?",
        "expected_keywords": ["single"],
        "expected_file": "charbel.txt",
        "expected_page": "N/A",
    },    
    {
        "question": "What are Charbel's hobbies?",
        "expected_keywords": ["basketball", "badminton", "pingpon", "magic", "bartending", "camping","Solving puzzles", "playing strategy board games"],
        "expected_file": "charbel.txt",
        "expected_page": "N/A",
    },
    {
        "question": "What languages does Charbel speak?",
        "expected_keywords": ["english", "french", "arabic",  "chinese"],
        "expected_file": "charbel.txt",
        "expected_page": "N/A",
    },
        {
        "question": "What is the level of french does Charbel speak?",
        "expected_keywords": ["Delf B2"],
        "expected_file": "charbel.txt",
        "expected_page": "N/A",
    },
    {
        "question": "What are Charbel's main AI interests?",
        "expected_keywords": ["deep learning", "reinforcement", "robotics","Rubik's cube"],
        "expected_file": "lebbous.docx",
        "expected_page": "N/A",
    },
    {
        "question": "Where did Charbel study Machine Learning?",
        "expected_keywords": ["lebanese university","lebanon"],
        "expected_file": "lebbous.docx",
        "expected_page": "N/A",
    },
    {
        "question": "What is the university degree charbel reach?",
        "expected_keywords": ["Master M1", "Master M2","M1","M2"],
        "expected_file": "lebbous.docx",
        "expected_page": "N/A",
    },
    {
        "question": "what is the email of charbel?",
        "expected_keywords": ["charbellebbousf@gmail.com"],
        "expected_file": "Charbel Lebbous CV.pdf",
        "expected_page": 0,
    },       
    {
        "question": "where does phoenix alliance operate?",
        "expected_keywords": ["europe", "mediterranean","france", "lebanon", "greece","morocco"],
        "expected_file": "[EN] PA - FY23 - Company Presentation.pdf",
        "expected_page": 1,
    },    
    {
        "question": "what is ghadir's role in the coompany?",
        "expected_keywords": [" expert of Microsoft collaborative solutions", " Scrum Master", "Projects Director"],
        "expected_file": "[EN] PA - FY23 - Company Presentation.pdf",
        "expected_page": 3,
    },
    {
        "question": "What is Aurélien's role in the company?",
        "expected_keywords": ["Microsoft Cloud Architect ", "former Microsoft employee"],
        "expected_file": "[EN] PA - FY23 - Company Presentation.pdf",
        "expected_page": 3,
    },
]

def test_pipeline(folder_path):
    docs = load_documents(folder_path)
    if not docs:
        print("❌ No documents loaded.")
        return

    indexer = EmbeddingIndexer()
    indexer.build_index(docs)

    passed_cases = 0
    total_keywords = 0
    total_keywords_found = 0
    correct_file_hits = 0
    correct_page_hits = 0
    pdf_cases = 0

    for test in test_cases:
        print(f"\n🧪 Q: {test['question']}")
        retrieved_docs = indexer.search(test["question"])
        answer = answer_question(test["question"], retrieved_docs)
        answer_text = answer.lower()

        # Keyword checking
        found_keywords = [kw for kw in test["expected_keywords"] if kw.lower() in answer_text]
        total_keywords += len(test["expected_keywords"])
        total_keywords_found += len(found_keywords)

        # File/Page match checking
        file_match = False
        page_match = False
        for doc in retrieved_docs:
            if doc.metadata.get("file_name") == test["expected_file"]:
                file_match = True
                if doc.metadata.get("page") == test["expected_page"]:
                    page_match = True
                break

        if file_match:
            correct_file_hits += 1
        if test["expected_page"] != "N/A":
            pdf_cases += 1
            if page_match:
                correct_page_hits += 1

        # Pass if at least one keyword found and file is correct
        passed = file_match and len(found_keywords) >= 1
        if passed:
            passed_cases += 1

        print(f"💬 Answer: {answer}")
        print(f"✅ Keywords Found: {found_keywords}")
        print(f"📄 File Match: {'✅' if file_match else '❌'} | Page Match: {'✅' if page_match else '❌'}")
        print(f"🟢 {'PASS' if passed else 'FAIL'}")

    print("\n📊 Performance Summary")
    print(f"✔️ Passed: {passed_cases}/{len(test_cases)} → {100 * passed_cases / len(test_cases):.2f}%")
    print(f"🧠 Keyword Recall: {total_keywords_found}/{total_keywords} → {100 * total_keywords_found / total_keywords:.2f}%")
    print(f"📄 File Match Rate: {correct_file_hits}/{len(test_cases)} → {100 * correct_file_hits / len(test_cases):.2f}%")
    if pdf_cases > 0:
        print(f"📑 Page Match Rate: {correct_page_hits}/{pdf_cases} → {100 * correct_page_hits / pdf_cases:.2f}%")

if __name__ == "__main__":
    folder = "./data"
    test_pipeline(folder)
