import asyncio
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from unittest.mock import AsyncMock, MagicMock
from ip_compression import DocumentProcessor

async def main():
    print("🚀 지능형 문서 전처리기(DocumentProcessor) 초기화 중...")
    processor = DocumentProcessor()

    mock_request = MagicMock()
    mock_request.is_disconnected = AsyncMock(return_value=False)

    test_file_path = r"C:\Users\happy\OneDrive\바탕 화면\Code\01_RAG\06_Preprocess\docs\[제논]GenOS_Flowise_교육자료.pdf"  # 예: "sample_document.hwp" 도 가능

    if not os.path.exists(test_file_path):
        print(f"\n⚠️ 주의: '{test_file_path}' 파일을 찾을 수 없습니다.")
        print("실제로 테스트를 진행하시려면 존재하는 PDF나 HWP 파일 경로로 코드를 수정해 주세요.")
        print("테스트 스크립트의 구조만 보여드리고 종료합니다.\n")
        return

    from dotenv import load_dotenv
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # [STEP 1] 전처리 및 Batch API 요약 요청 전송
    print(f"\n📄 [STEP 1] '{test_file_path}' 문서 처리를 시작합니다...")
    try:
        vectors = await processor(
            request=mock_request,
            file_path=test_file_path,
            log_level=4,                
            max_chunk_size=1024,        
            auto_convert_to_pdf=True,   
            use_pdf_sdk=False,          
            appendix="[]",              
            # --- [NEW] Batch API 모드 활성화 ---
            inject_summary="batch" if OPENAI_API_KEY else False
        )
        
        print(f"\n✅ 전처리 완료! 총 {len(vectors)}개의 기본 청크(Vector)가 생성되었습니다.")
        
        # Batch 모드일 경우 메타데이터에 pending_batch_id가 발급됩니다.
        batch_id = vectors[0].pending_batch_id if len(vectors) > 0 and hasattr(vectors[0], "pending_batch_id") else None
        
        if batch_id:
            print(f"\n⏳ [대기 상태] OpenAI Batch 요약 작업이 서버에 등록되었습니다!")
            print(f"👉 Batch ID: {batch_id}")
            print("최대 24시간 내에 백그라운드에서 처리가 완료되며, 완료 시 요약 주입이 가능합니다.")
            
            # [STEP 2] 나중에 결과를 확인하고 주입하는 방법 (예제)
            check_later = input("\n결과를 즉시 확인해 보시겠습니까? (완료되지 않았으면 실패합니다) (y/n): ")
            if check_later.lower() == 'y':
                from ip_modules.batch_utils import check_batch_status, retrieve_batch_results
                
                status = await check_batch_status(batch_id)
                print(f"현재 Batch 상태: {status}")
                
                if status == "completed":
                    # 요약본을 가져와서 기존 벡터에 주입합니다.
                    summaries_map = await retrieve_batch_results(batch_id)
                    for chunk_idx, summary in summaries_map.items():
                        if chunk_idx < len(vectors):
                            vectors[chunk_idx].text = f"[문서 요약: {summary}]\n\n" + vectors[chunk_idx].text
                    
                    print("\n🎉 요약 주입이 완료되었습니다!")
                    print(f"첫 번째 청크 텍스트: {vectors[0].text[:100]}...")
                else:
                    print("아직 완료되지 않았습니다. 나중에 다시 확인해 주세요.")
        else:
            print("\nBatch 모드가 아니거나 API Key가 없어 원본 청크만 출력합니다.")

        # ==============================================================================
        # [STEP 3] 청킹된 결과물 파일로 저장 (저장 로직 추가)
        # ==============================================================================
        import json
        output_file_name = "chunked_result.json"
        
        # Pydantic 모델(GenOSVectorMeta) 리스트를 순회하며 dict로 변환하여 저장합니다.
        # getattr를 사용해 dict 형태와 Pydantic 객체 형태 모두 호환되도록 처리합니다.
        vectors_dict_list = []
        for v in vectors:
            if hasattr(v, "model_dump"):
                vectors_dict_list.append(v.model_dump())
            elif hasattr(v, "dict"):
                vectors_dict_list.append(v.dict())
            elif isinstance(v, dict):
                vectors_dict_list.append(v)
            else:
                vectors_dict_list.append(v.__dict__)

        with open(output_file_name, "w", encoding="utf-8") as f:
            json.dump(vectors_dict_list, f, ensure_ascii=False, indent=2)
            
        print(f"\n💾 청킹된 전체 문서 데이터가 '{output_file_name}' 파일로 성공적으로 저장되었습니다!")
        print("해당 파일을 열어보시면 청킹된 모든 텍스트와 메타데이터(페이지, Bbox 등)를 확인하실 수 있습니다.")

    except Exception as e:
        print(f"\n❌ 전처리 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    asyncio.run(main())
