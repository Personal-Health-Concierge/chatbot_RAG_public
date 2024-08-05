import os
import boto3
import genai_core.types
import genai_core.chunks
import genai_core.documents
import genai_core.workspaces
import genai_core.aurora.create
from langchain.document_loaders import S3FileLoader


import fitz  # PyMuPdf
import base64
from openai import OpenAI, OpenAIError

os.environ["OPENAI_API_KEY"] = ""

def convert_pdf_to_image(pdf_file):
    image_files = []
    if pdf_file.endswith(".pdf"):
        base_name = os.path.basename(pdf_file)
        base_name = base_name[:-4]
        doc = fitz.open(pdf_file)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image = page.get_pixmap()
            image_file_name = f"/tmp/{base_name}_{page_num + 1}.png"
            image.save(image_file_name)
            image_files.append(image_file_name)
        doc.close()
    return image_files


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def summarize_image_inputs(images):
    if (not images) or (len(images) <= 0):
        return
    content_arr = []
    content_arr.append(
        {
            "type": "text",
            "text": "Extract and summarize health related information from the following image files for the most recent date",
        }
    )
    for image in images:
        base64_image = encode_image(image)
        content_arr.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            }
        )
    try:
        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": content_arr,
                }
            ],
            temperature=0.0,
            max_tokens=4095,
            top_p=0.9,
        )
        return completion.choices[0].message.content
    except OpenAIError as e:
        # Handle all OpenAI API errors
        return f"Error: {e}"


def process_pdf(pdf_file):
    images = convert_pdf_to_image(pdf_file)
    result = summarize_image_inputs(images)
    return result


WORKSPACE_ID = os.environ.get("WORKSPACE_ID")
DOCUMENT_ID = os.environ.get("DOCUMENT_ID")
INPUT_BUCKET_NAME = os.environ.get("INPUT_BUCKET_NAME")
INPUT_OBJECT_KEY = os.environ.get("INPUT_OBJECT_KEY")
PROCESSING_BUCKET_NAME = os.environ.get("PROCESSING_BUCKET_NAME")
PROCESSING_OBJECT_KEY = os.environ.get("PROCESSING_OBJECT_KEY")

s3_client = boto3.client("s3")


def main():
    print("Starting file converter batch job")
    print("Workspace ID: {}".format(WORKSPACE_ID))
    print("Document ID: {}".format(DOCUMENT_ID))
    print("Input bucket name: {}".format(INPUT_BUCKET_NAME))
    print("Input object key: {}".format(INPUT_OBJECT_KEY))
    print("Output bucket name: {}".format(PROCESSING_BUCKET_NAME))
    print("Output object key: {}".format(PROCESSING_OBJECT_KEY))

    workspace = genai_core.workspaces.get_workspace(WORKSPACE_ID)
    if not workspace:
        raise genai_core.types.CommonError(f"Workspace {WORKSPACE_ID} does not exist")

    document = genai_core.documents.get_document(WORKSPACE_ID, DOCUMENT_ID)
    if not document:
        raise genai_core.types.CommonError(
            f"Document {WORKSPACE_ID}/{DOCUMENT_ID} does not exist"
        )

    try:
        extension = os.path.splitext(INPUT_OBJECT_KEY)[-1].lower()
        if extension == ".txt":
            object = s3_client.get_object(
                Bucket=INPUT_BUCKET_NAME, Key=INPUT_OBJECT_KEY
            )
            content = object["Body"].read().decode("utf-8")
        elif extension == ".pdf":
            pdf_file = "/tmp/from_s3.pdf"
            s3_client.download_file(INPUT_BUCKET_NAME, INPUT_OBJECT_KEY, pdf_file)
            content = process_pdf(pdf_file)
        else:
            loader = S3FileLoader(INPUT_BUCKET_NAME, INPUT_OBJECT_KEY)
            print(f"loader: {loader}")
            docs = loader.load()
            content = docs[0].page_content

        if (
            INPUT_BUCKET_NAME != PROCESSING_BUCKET_NAME
            and INPUT_OBJECT_KEY != PROCESSING_OBJECT_KEY
        ):
            s3_client.put_object(
                Bucket=PROCESSING_BUCKET_NAME, Key=PROCESSING_OBJECT_KEY, Body=content
            )

        add_chunks(workspace, document, content)
    except Exception as error:
        genai_core.documents.set_status(WORKSPACE_ID, DOCUMENT_ID, "error")
        print(error)
        raise error


def add_chunks(workspace: dict, document: dict, content: str):
    chunks = genai_core.chunks.split_content(workspace, content)

    genai_core.chunks.add_chunks(
        workspace=workspace,
        document=document,
        document_sub_id=None,
        chunks=chunks,
        chunk_complements=None,
        replace=True,
    )


if __name__ == "__main__":
    main()
