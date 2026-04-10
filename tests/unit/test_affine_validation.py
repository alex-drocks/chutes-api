import textwrap
from api.affine import check_affine_code, force_affine_engine_args


def assert_valid(code: str) -> None:
    valid, message = check_affine_code(textwrap.dedent(code))
    assert valid, message


def assert_invalid(code: str, expected_substring: str) -> None:
    valid, message = check_affine_code(textwrap.dedent(code))
    assert not valid, "expected invalid code"
    assert expected_substring in message


def test_valid_sglang_chute_with_engine_args() -> None:
    assert_valid(
        """
        from chutes.chute import NodeSelector
        from chutes.chute.template.sglang import build_sglang_chute

        chute = build_sglang_chute(
            username="exampleuser",
            readme="foo/affine-test",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            concurrency=40,
            revision="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            node_selector=NodeSelector(
                gpu_count=1,
                min_vram_gb_per_gpu=80,
            ),
            max_instances=5,
            shutdown_after_seconds=28800,
            scaling_threshold=0.5,
            engine_args=(
                "--mem-fraction-static 0.8 "
                "--context-length 36384"
            ),
        )
        """
    )


def test_valid_vllm_chute_with_engine_args() -> None:
    assert_valid(
        """
        from chutes.chute import NodeSelector
        from chutes.chute.template.vllm import build_vllm_chute

        chute = build_vllm_chute(
            username="exampleuser",
            readme="foo/affine-test",
            model_name="foo/affine-test",
            image="chutes/vllm:nightly-2026010900",
            concurrency=40,
            revision="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            node_selector=NodeSelector(
                gpu_count=1,
                min_vram_gb_per_gpu=80,
            ),
            max_instances=5,
            shutdown_after_seconds=28800,
            scaling_threshold=0.5,
            engine_args=(
                "--mem-fraction-static 0.8 "
                "--context-length 36384"
            ),
        )
        """
    )


def test_valid_allowed_env_vars() -> None:
    assert_valid(
        """
        import os
        from chutes.chute.template.sglang import build_sglang_chute

        os.environ["VLLM_BATCH_INVARIANT"] = "1"

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            engine_args="--context-length 4096",
        )
        """
    )


def test_invalid_disallowed_env_var() -> None:
    assert_invalid(
        """
        import os
        from chutes.chute.template.sglang import build_sglang_chute

        os.environ["AWS_SECRET_ACCESS_KEY"] = "nope"

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            engine_args="--context-length 4096",
        )
        """,
        "Setting os.environ['AWS_SECRET_ACCESS_KEY'] is not allowed",
    )


def test_invalid_engine_args_trust_remote_code() -> None:
    assert_invalid(
        """
        from chutes.chute.template.sglang import build_sglang_chute

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            engine_args="--trust-remote-code true",
        )
        """,
        "engine_args cannot contain 'trust_remote_code' or 'trust-remote-code'",
    )


def test_invalid_engine_args_concatenated_flags() -> None:
    assert_invalid(
        """
        from chutes.chute.template.vllm import build_vllm_chute

        chute = build_vllm_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/vllm:nightly-2026010900",
            engine_args="--mem-fraction-static 0.8--context-length 36384",
        )
        """,
        "engine_args appears to contain concatenated flags",
    )


def test_invalid_engine_args_mem_fraction_static_too_high() -> None:
    assert_invalid(
        """
        from chutes.chute.template.sglang import build_sglang_chute

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            engine_args="--mem-fraction-static 0.9 --context-length 36384",
        )
        """,
        "--mem-fraction-static value 0.9 is too high (must be < 0.85",
    )


def test_invalid_engine_args_mem_fraction_static_at_boundary() -> None:
    assert_invalid(
        """
        from chutes.chute.template.vllm import build_vllm_chute

        chute = build_vllm_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/vllm:nightly-2026010900",
            engine_args="--mem-fraction-static 0.85",
        )
        """,
        "--mem-fraction-static value 0.85 is too high (must be < 0.85",
    )


def test_invalid_engine_args_gpu_memory_utilization_too_high() -> None:
    assert_invalid(
        """
        from chutes.chute.template.vllm import build_vllm_chute

        chute = build_vllm_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/vllm:nightly-2026010900",
            engine_args="--gpu-memory-utilization 0.95",
        )
        """,
        "--gpu-memory-utilization value 0.95 is too high (must be < 0.85",
    )


def test_invalid_engine_args_non_string() -> None:
    assert_invalid(
        """
        from chutes.chute.template.vllm import build_vllm_chute

        chute = build_vllm_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/vllm:nightly-2026010900",
            engine_args={"foo": "bar"},
        )
        """,
        "engine_args for build_vllm_chute must be a string literal",
    )


def test_invalid_image_prefix_for_builder() -> None:
    assert_invalid(
        """
        from chutes.chute.template.vllm import build_vllm_chute

        chute = build_vllm_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            engine_args="--context-length 4096",
        )
        """,
        "image must start with 'chutes/vllm'",
    )


def test_invalid_import_outside_chutes() -> None:
    assert_invalid(
        """
        import math
        from chutes.chute.template.sglang import build_sglang_chute

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            engine_args="--context-length 4096",
        )
        """,
        "Invalid import: math",
    )


def test_invalid_missing_chute_assignment() -> None:
    assert_invalid(
        """
        from chutes.chute.template.sglang import build_sglang_chute

        something_else = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            engine_args="--context-length 4096",
        )
        """,
        "Function build_sglang_chute must be assigned to variable 'chute'",
    )


def test_invalid_engine_args_f_string_obfuscation() -> None:
    assert_invalid(
        """
        from chutes.chute.template.sglang import build_sglang_chute

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            engine_args=f"--trust-remote-code true",
        )
        """,
        "f-strings are not allowed",
    )


def test_invalid_engine_args_format_obfuscation() -> None:
    assert_invalid(
        """
        from chutes.chute.template.vllm import build_vllm_chute

        chute = build_vllm_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/vllm:nightly-2026010900",
            engine_args="--trust-{0} true".format("remote-code"),
        )
        """,
        "Dangerous function 'format' is not allowed",
    )


def test_invalid_engine_args_join_obfuscation() -> None:
    assert_invalid(
        """
        from chutes.chute.template.vllm import build_vllm_chute

        chute = build_vllm_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/vllm:nightly-2026010900",
            engine_args="".join(["--trust-remote", "-code true"]),
        )
        """,
        "String .join() method is not allowed",
    )


def test_invalid_engine_args_concat_non_literal() -> None:
    assert_invalid(
        """
        from chutes.chute.template.sglang import build_sglang_chute

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            engine_args="--trust-" + "remote-code true",
        )
        """,
        "engine_args for build_sglang_chute must be a string literal",
    )


def test_invalid_engine_args_hex_decode_obfuscation() -> None:
    assert_invalid(
        """
        from chutes.chute.template.sglang import build_sglang_chute

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            engine_args=bytes.fromhex("2d2d74727573742d72656d6f74652d636f64652074727565"),
        )
        """,
        "engine_args for build_sglang_chute must be a string literal",
    )


def test_invalid_env_var_contains_trust() -> None:
    assert_invalid(
        """
        import os
        from chutes.chute.template.vllm import build_vllm_chute

        os.environ["VLLM_TRUST_REMOTE_CODE"] = "1"

        chute = build_vllm_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/vllm:nightly-2026010900",
            engine_args="--context-length 4096",
        )
        """,
        "Setting os.environ['VLLM_TRUST_REMOTE_CODE'] is not allowed",
    )


def test_invalid_os_environ_update() -> None:
    assert_invalid(
        """
        import os
        from chutes.chute.template.sglang import build_sglang_chute

        os.environ.update({"VLLM_BATCH_INVARIANT": "1"})

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            engine_args="--context-length 4096",
        )
        """,
        "os.environ.update is not allowed",
    )


def test_invalid_os_environ_setdefault() -> None:
    assert_invalid(
        """
        import os
        from chutes.chute.template.vllm import build_vllm_chute

        os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")

        chute = build_vllm_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/vllm:nightly-2026010900",
            engine_args="--context-length 4096",
        )
        """,
        "os.environ.setdefault is not allowed",
    )


# --- force_affine_engine_args tests ---


def test_force_sglang_adds_missing_engine_args() -> None:
    code = textwrap.dedent("""
        from chutes.chute.template.sglang import build_sglang_chute

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
        )
    """)
    result = force_affine_engine_args(code)
    assert result is not None
    assert "--mem-fraction-static 0.8" in result
    assert "--chunked-prefill-size 4096" in result


def test_force_vllm_adds_missing_engine_args() -> None:
    code = textwrap.dedent("""
        from chutes.chute.template.vllm import build_vllm_chute

        chute = build_vllm_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/vllm:nightly-2026010900",
        )
    """)
    result = force_affine_engine_args(code)
    assert result is not None
    assert "--gpu-memory-utilization 0.8" in result
    assert "--max-num-batched-tokens 4096" in result


def test_force_replaces_conflicting_numeric_values() -> None:
    code = textwrap.dedent("""
        from chutes.chute.template.sglang import build_sglang_chute

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            engine_args="--mem-fraction-static 0.95 --chunked-prefill-size 2048",
        )
    """)
    result = force_affine_engine_args(code)
    assert result is not None
    assert "--mem-fraction-static 0.8" in result
    assert "--chunked-prefill-size 4096" in result
    assert "0.95" not in result
    assert "2048" not in result


def test_force_replaces_non_numeric_values() -> None:
    code = textwrap.dedent("""
        from chutes.chute.template.vllm import build_vllm_chute

        chute = build_vllm_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/vllm:nightly-2026010900",
            engine_args="--gpu-memory-utilization auto --max-num-batched-tokens auto",
        )
    """)
    result = force_affine_engine_args(code)
    assert result is not None
    assert "--gpu-memory-utilization 0.8" in result
    assert "--max-num-batched-tokens 4096" in result
    assert "auto" not in result


def test_force_preserves_other_flags() -> None:
    code = textwrap.dedent("""
        from chutes.chute.template.sglang import build_sglang_chute

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            engine_args="--context-length 36384 --mem-fraction-static 0.8 --chunked-prefill-size 4096",
        )
    """)
    result = force_affine_engine_args(code)
    # No change needed — should return original code.
    assert result == code
    assert "--context-length" in result


def test_force_returns_none_for_no_builder() -> None:
    code = textwrap.dedent("""
        x = 1 + 2
    """)
    assert force_affine_engine_args(code) is None


def test_force_returns_none_for_syntax_error() -> None:
    assert force_affine_engine_args("def (((") is None


def test_force_no_duplicate_flags() -> None:
    code = textwrap.dedent("""
        from chutes.chute.template.vllm import build_vllm_chute

        chute = build_vllm_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/vllm:nightly-2026010900",
            engine_args="--gpu-memory-utilization 0.7 --max-num-batched-tokens 2048 --context-length 4096",
        )
    """)
    result = force_affine_engine_args(code)
    assert result is not None
    assert result.count("--gpu-memory-utilization") == 1
    assert result.count("--max-num-batched-tokens") == 1


def test_force_targets_top_level_assignment_only() -> None:
    """Only the top-level assignment to a builder call should be rewritten."""
    code = textwrap.dedent("""
        from chutes.chute.template.sglang import build_sglang_chute

        chute = build_sglang_chute(
            username="exampleuser",
            model_name="foo/affine-test",
            image="chutes/sglang:nightly-2025121000",
            engine_args="--context-length 36384",
        )
    """)
    result = force_affine_engine_args(code)
    assert result is not None
    assert "--mem-fraction-static 0.8" in result
    assert "--chunked-prefill-size 4096" in result
    # The original flag is preserved.
    assert "--context-length" in result
