from galaxies_tool.video import tipo_de_erro


def test_video_removido_e_permanente():
    assert tipo_de_erro("ERROR: [youtube] T6uAviiuAjo: This video is not available") == "permanente"
    assert tipo_de_erro("ERROR: [youtube] abc: Private video. Sign in if you've been granted access") == "permanente"


def test_verificacao_de_robo_e_bloqueio():
    assert tipo_de_erro("ERROR: [youtube] tn9EMIJeCWI: Sign in to confirm you’re not a bot.") == "bloqueio"


def test_403_rede_e_desconhecido_sao_temporarios():
    assert tipo_de_erro("ERROR: unable to download video data: HTTP Error 403: Forbidden") == "temporario"
    assert tipo_de_erro("ERROR: [download] Got error: ... [Errno 61] Connection refused.") == "temporario"
    assert tipo_de_erro("ERROR: [youtube] 96AZuQ8C0mA: Please sign in.") == "temporario"
    assert tipo_de_erro("") == "temporario"
