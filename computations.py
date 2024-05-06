import requests
import base64


def generate_img(api_key, prompt, output_img_path):
    engine_id = "stable-diffusion-v3"

    if engine_id == "stable-diffusion-v1-6":
        api_host = 'https://api.stability.ai'

        response = requests.post(
            f"{api_host}/v1/generation/{engine_id}/text-to-image",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            json={
                "text_prompts": [
                    {
                        "text": prompt
                    }
                ],
                "cfg_scale": 7,
                "height": 512,
                "width": 512,
                "samples": 1,
                "steps": 30,
            },
        )

        if response.status_code != 200:
            raise Exception("Non-200 response: " + str(response.text))

        data = response.json()

        path = ""
        for i, image in enumerate(data["artifacts"]):
            path = f"{output_img_path.split('.')[0]}_{i}.{output_img_path.split('.')[1]}"
            with open(path, "wb") as f:
                f.write(base64.b64decode(image["base64"]))
        return path

    elif engine_id == "stable-diffusion-v3":
        response = requests.post(
            f"https://api.stability.ai/v2beta/stable-image/generate/sd3",
            headers={
                "authorization": f"Bearer {api_key}",
                "accept": "image/*"
            },
            files={
                "none": ''
            },
            data={
                "prompt": prompt,
                "output_format": "png"
            },
        )


        if response.status_code == 200:
            with open(output_img_path, 'wb') as file:
                file.write(response.content)
        else:
            raise Exception(str(response.json()))

        return output_img_path

    else:
        raise Exception(f"Invalid engine_id: {engine_id}")


def compute(api_key, prompt):
    """
    Generates images based on an input prompt using the Stability AI API.
    Input:
        api_key: your Stability AI API key.
        prompt: the input prompt.
    Output:
        output_img: path to the generated image.
    """
    output_img_path = f"sd_generated_image.png"
    output_img = generate_img(api_key, prompt, output_img_path)

    return {"output_img": output_img}


def test():
    """Test the compute function."""

    print("Running test")
