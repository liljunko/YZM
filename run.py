from default_config import Config

if __name__ == "__main__":
    from captcha_rec import train

    config = Config()
    assert config.cuda, "Please don't train your code using only cpu"
    train(config)
