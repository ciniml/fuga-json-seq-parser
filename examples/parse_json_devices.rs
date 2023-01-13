use embedded_io::adapters;
use fuga_json_seq_parser::{DefaultParserCallbackResult, JsonNode, Parser, ParserCallbackAction};
use std::{fs::File, io::Read};

fn main() {
    let mut file = File::open("data/devices.json").unwrap();
    let mut parser: Parser<256, 10> = Parser::new();
    let mut reader = embedded_io::adapters::FromStd::new(&mut file);
    let mut indent_level = 0;
    parser
        .parse(&mut reader, |node| {
            match node {
                JsonNode::EndMap => indent_level -= 1,
                JsonNode::EndArray => indent_level -= 1,
                _ => {}
            }
            for _ in 0..indent_level {
                print!("  ");
            }
            match node {
                JsonNode::StartMap => println!("{{"),
                JsonNode::StartArray => println!("["),
                JsonNode::Key(v) => print!("{}: ", v),
                JsonNode::Value(v) => println!("{},", v),
                JsonNode::EndMap => println!("}},"),
                JsonNode::EndArray => println!("],"),
            }
            match node {
                JsonNode::StartMap => indent_level += 1,
                JsonNode::StartArray => indent_level += 1,
                _ => {}
            }
            DefaultParserCallbackResult::Ok(ParserCallbackAction::Nothing)
        })
        .unwrap();
}
